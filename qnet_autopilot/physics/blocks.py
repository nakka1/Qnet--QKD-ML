"""Where to cut the key blocks, when the error rate moves during the run.

Every finite-key figure in this package so far assumed blocks of one fixed
duration: the link accumulates sifted bits for a second, settles up, starts
again. That is a modelling convenience, and the reports flagged it as one. A
real receiver chooses when to close a block, and the choice matters for a
reason that is specific to the finite-key bound: **a block is scored on its own
average error rate, and a stretch of bad fibre poisons every bit in the block it
lands in**. Ten good seconds and one bad second in the same block can be worth
nothing; the same eleven seconds cut into two blocks are worth ten seconds of
key.

Three strategies, one interface
-------------------------------
:func:`fixed_blocks` cuts on a timer, which is what phases 3 and 4 assumed.
:func:`adaptive_blocks` is a controller: it watches a noisy estimate of the
error rate — the one a receiver actually has, from the bits it publishes for
parameter estimation — and at every decision instant compares two futures,
closing the block when closing is worth more than carrying it. It has no tuning
constant. :func:`optimal_blocks` is the oracle: dynamic programming over the
true error series, choosing the partition that maximises total key with perfect
hindsight.

The oracle is not a proposal. It exists so the controller can be judged against
the best any strategy could do rather than against the fixed-timer baseline it
is bound to beat. It also has to be the *right* oracle: an earlier version of
this study searched only over fixed durations, one per run, capped at ten
seconds, concluded that variable blocking bought 1.28x at most, and recommended
against building any of this. That oracle cannot express the one thing a
variable block does — cut the good window out of a bad run — and the conclusion
drawn from it was wrong. On a badly compensated link a timer of any duration
above a second yields exactly zero bits, and the controller here yields
hundreds.

A caveat this module does not close: the oracle partitions the run into
*intervals*. Privacy amplification does not require a block to be contiguous in
time, so a receiver could pool good windows from separate stretches into one
block, and that oracle — a prefix of the segments sorted by error rate — is
strictly larger than this one.

Why dynamic programming is exact here
-------------------------------------
The key extractable from a block depends only on where it starts and where it
ends — on the number of bits and their mean error rate, and on nothing outside
those bounds. So the total over a partition decomposes, and the best partition
of the first :math:`j` segments is the best over choices of where the last block
began. That is an :math:`O(n^2)` recursion with no approximation in it, which
is what makes the bound a real ceiling rather than a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .link import bb84_finite_key_length

__all__ = ["BlockPlan", "fixed_blocks", "adaptive_blocks", "optimal_blocks"]


@dataclass(frozen=True)
class BlockPlan:
    """One way of cutting a run into key blocks, and what it yielded.

    Blocks are half-open ``[start, end)`` sample ranges and need not cover the
    run: a controller may **discard** a stretch it judges unusable rather than
    fold it into a block, which is the whole point of adapting. The rate is
    always divided by the length of the run, not by the time spent inside
    blocks, so discarding never flatters a strategy.

    Attributes:
        starts, ends: sample indices bounding each block.
        key_bits: secret bits extracted from each block.
        dt_s: sampling interval of the error series the plan was built on.
        run_samples: length of the whole run, the denominator for the rate.
        strategy: which rule produced it, for reporting.
    """

    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    key_bits: NDArray[np.float64]
    dt_s: float
    run_samples: int
    strategy: str = ""

    @property
    def count(self) -> int:
        return int(self.key_bits.size)

    @property
    def durations_s(self) -> NDArray[np.float64]:
        return (self.ends - self.starts) * self.dt_s

    @property
    def total_key_bits(self) -> float:
        return float(np.sum(self.key_bits))

    @property
    def duration_s(self) -> float:
        return float(self.run_samples * self.dt_s)

    @property
    def mean_key_rate_hz(self) -> float:
        """Secret bits per second over the whole run, discarded time included."""
        return self.total_key_bits / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def duty(self) -> float:
        """Share of the run that sat inside some block."""
        if self.run_samples <= 0:
            return 0.0
        return float(np.sum(self.ends - self.starts) / self.run_samples)

    @property
    def empty_fraction(self) -> float:
        """Share of blocks that yielded nothing.

        Worth reporting on its own: a strategy can have a good mean and still be
        throwing away half its blocks, and those are the seconds during which a
        deployed link is producing no key at all.
        """
        return float(np.mean(self.key_bits <= 0.0)) if self.count else 0.0


def _segment_edges(samples: int, dt_s: float, granularity_s: float
                   ) -> NDArray[np.int64]:
    """Candidate block boundaries, on a coarser grid than the samples.

    A block may only start and end on one of these. The grid is what keeps the
    dynamic programme quadratic in a few hundred rather than in tens of
    thousands, and it costs nothing real: no receiver closes a block on a
    microsecond boundary either.
    """
    stride = max(int(round(granularity_s / dt_s)), 1)
    edges = np.arange(0, samples + 1, stride, dtype=np.int64)
    if edges[-1] != samples:
        edges = np.append(edges, samples)
    return edges


def _block_key_bits(prefix: NDArray[np.float64], starts: ArrayLike,
                    ends: ArrayLike, sifted_rate_hz: float, dt_s: float,
                    **finite_key_kwargs) -> NDArray[np.float64]:
    """Key from each ``[start, end)`` block, vectorised over the pairs.

    Args:
        prefix: cumulative sum of the error series, with a leading zero, so a
            block's mean error is one subtraction.
        starts, ends: sample indices, broadcast against each other.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        **finite_key_kwargs: passed to
            :func:`~qnet_autopilot.physics.link.bb84_finite_key_length`.

    Returns:
        Secret bits per block, never negative.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    lengths = (ends - starts).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_qber = np.where(lengths > 0,
                             (prefix[ends] - prefix[starts]) / np.maximum(lengths, 1),
                             1.0)
    block_bits = sifted_rate_hz * lengths * dt_s
    return bb84_finite_key_length(block_bits, mean_qber, **finite_key_kwargs)


def fixed_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                 duration_s: float, **finite_key_kwargs) -> BlockPlan:
    """Cut on a timer, which is what phases 3 and 4 assumed.

    Args:
        qber: error rate per sample.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        duration_s: block length. A trailing partial block is dropped rather
            than scored short, because a receiver would carry it into the next
            block rather than settle up on it.

    Returns:
        The plan.

    Raises:
        ValueError: on a non-positive duration or interval.
    """
    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("duration_s and dt_s must be positive")

    error = np.asarray(qber, dtype=float)
    per_block = max(int(round(duration_s / dt_s)), 1)
    count = error.size // per_block
    if count == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, error.size,
                         strategy=f"fixo {duration_s:g} s")

    boundaries = np.arange(count + 1, dtype=np.int64) * per_block
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    key_bits = _block_key_bits(prefix, boundaries[:-1], boundaries[1:],
                               sifted_rate_hz, dt_s, **finite_key_kwargs)
    return BlockPlan(boundaries[:-1], boundaries[1:], key_bits, dt_s,
                     error.size, strategy=f"fixo {duration_s:g} s")


def optimal_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                   granularity_s: float = 0.1,
                   max_block_s: float | None = None,
                   **finite_key_kwargs) -> BlockPlan:
    """The partition that maximises total key, with perfect hindsight.

    Args:
        qber: error rate per sample — the *true* one, which is what makes this
            an oracle and not a proposal.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        granularity_s: candidate boundaries are on this grid.
        max_block_s: cap on block length, if the application has one.
        **finite_key_kwargs: passed through to the finite-key bound.

    Returns:
        The plan. Its total is an upper bound on every strategy that cuts on
        this grid, causal or not.

    Note:
        The recursion is exact because a block's key depends only on its own
        endpoints. That is a property of the finite-key formula, not an
        assumption: privacy amplification is applied per block, and no term
        couples one block to another.
    """
    error = np.asarray(qber, dtype=float)
    if error.size == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, 0, strategy="oráculo")

    edges = _segment_edges(error.size, dt_s, granularity_s)
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    segments = edges.size

    maximum_samples = (error.size if max_block_s is None
                       else max(int(round(max_block_s / dt_s)), 1))

    best = np.full(segments, -np.inf)
    best[0] = 0.0
    previous = np.zeros(segments, dtype=np.int64)

    for end in range(1, segments):
        reachable = edges[end] - edges[:end] <= maximum_samples
        candidates = np.flatnonzero(reachable)
        if candidates.size == 0:
            candidates = np.array([end - 1])
        gains = _block_key_bits(prefix, edges[candidates],
                                np.full(candidates.size, edges[end]),
                                sifted_rate_hz, dt_s, **finite_key_kwargs)
        totals = best[candidates] + gains
        winner = int(np.argmax(totals))
        best[end] = totals[winner]
        previous[end] = candidates[winner]

    boundaries = [segments - 1]
    while boundaries[-1] != 0:
        boundaries.append(int(previous[boundaries[-1]]))
    indices = np.array(sorted(boundaries), dtype=np.int64)
    cut = edges[indices]
    key_bits = _block_key_bits(prefix, cut[:-1], cut[1:], sifted_rate_hz, dt_s,
                               **finite_key_kwargs)
    # A block the recursion left empty is a stretch it chose to write off, and
    # reporting it as a block would understate the duty cycle of the ones that
    # actually produced. Dropping them changes no total.
    keep = key_bits > 0.0
    if not np.any(keep):
        keep = np.ones_like(key_bits, dtype=bool)
    return BlockPlan(cut[:-1][keep], cut[1:][keep], key_bits[keep], dt_s,
                     error.size, strategy="oráculo")


def adaptive_blocks(qber: ArrayLike, sifted_rate_hz: float, dt_s: float,
                    granularity_s: float = 0.1,
                    estimation_fraction: float = 0.1,
                    abort_qber: float = 0.11,
                    max_block_s: float = 20.0,
                    rng: np.random.Generator | int | None = None,
                    **finite_key_kwargs) -> BlockPlan:
    """A causal controller: close a block when closing is worth more than not.

    The obvious rule — close when the newest segment would lower the block's
    key — is myopic and measurably wrong. It closes on any excursion, and each
    close makes the next block pay the finite-size cost over again: on a
    well-compensated link it gives up about a fifth of the key by cutting into
    two-second blocks where one twenty-second block was available. What the
    controller has to compare is not two blocks but two *futures*.

    So it plans. At every decision instant it knows three things, all of them
    from bits the protocol has already published for parameter estimation:

    * ``q_block`` — the error rate of the block it is holding open;
    * ``q_segment`` — the error rate of the segment just measured;
    * ``q_link`` — the error rate of the link over the run so far, which is its
      forecast of what the remaining fibre will look like.

    From ``q_link`` it computes ``R``, the key rate a stationary link of that
    quality would sustain under its own block budget, and then values the two
    options over the horizon the budget defines:

    * **close now** — bank the block as it stands, and run at ``R`` for the
      rest of the horizon;
    * **keep it open** — carry the block, this segment included, to whichever
      later closing point is worth most, and run at ``R`` after that.

    Both branches then behave correctly without a tuning constant:

    * **A quiet link is left alone.** ``R`` is high, so throwing away a
      part-grown block to restart at the same quality gains nothing, and the
      controller extends. It converges on one long block, which is what the
      oracle does too, and a single noisy segment cannot talk it out of that
      because the option value is taken over the *best* later closing point,
      not over closing at once.
    * **A bad link is cut and skipped.** When ``q_link`` is above the
      threshold, ``R`` is zero: the future is worth nothing, and the only value
      on the table is the block already in hand. Any segment that would lower
      it closes it, and a segment estimated above ``abort_qber`` opens no new
      block, so the bad stretch is discarded rather than averaged in. On a link
      that spends most of its time above the threshold, that is the difference
      between some key and none — no fixed timer, of any duration, produces a
      single bit there.

    Args:
        qber: the true error rate per sample. The controller never sees it —
            it sees the binomial draws taken from it below.
        sifted_rate_hz: sifted bits per second.
        dt_s: sampling interval.
        granularity_s: how often the controller may act, and the grid the
            planned closing points are searched on.
        estimation_fraction: share of the sifted bits published as they are
            produced, which is what the controller estimates from. This is the
            *monitoring* budget, not the test fraction the bound optimises
            internally: a receiver discloses a running subset to watch the link
            and settles the accounting at the end of the block. The noise is
            the difficulty — a segment that looks bad may only have been
            unlucky, and closing on it costs real key.
        abort_qber: a segment estimated above this cannot contribute to any
            block, so none is opened on it. The BB84 threshold is the natural
            value.
        max_block_s: the block budget — the longest a block may stay open, and
            therefore also the horizon the controller plans over. A deployment
            with a key-delivery deadline sets it from that deadline.
        rng: seeds the estimation noise.
        **finite_key_kwargs: passed through to the finite-key bound, both for
            the controller's own forecast and for the scoring.

    Returns:
        The plan. Blocks need not tile the run; discarded stretches are absent
        from it and still count in the denominator of the rate.

    Note:
        The controller decides on estimates and is scored on the truth. A rule
        judged on the same noisy numbers it acts on would hide the cost of
        reacting to noise, which is most of what separates a controller from an
        oracle.
    """
    error = np.asarray(qber, dtype=float)
    if error.size == 0:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, 0, strategy="adaptativo")
    if max_block_s <= 0 or granularity_s <= 0 or dt_s <= 0:
        raise ValueError("max_block_s, granularity_s and dt_s must be positive")

    generator = np.random.default_rng(rng)
    edges = _segment_edges(error.size, dt_s, granularity_s)
    prefix = np.concatenate([[0.0], np.cumsum(error)])
    maximum_samples = max(int(round(max_block_s / dt_s)), 1)
    #: Bits published per sample of fibre time — the controller's whole view.
    disclosed_per_sample = sifted_rate_hz * dt_s * estimation_fraction

    def disclose(start: int, end: int) -> tuple[float, float]:
        """Bits published over ``[start, end)`` and how many of them erred."""
        samples = end - start
        if samples <= 0:
            return 0.0, 0.0
        truth = float(np.clip((prefix[end] - prefix[start]) / samples, 0.0, 1.0))
        tested = int(disclosed_per_sample * samples)
        if tested < 1:
            # Too few published bits to estimate from; the controller is blind
            # and can only take the segment at face value.
            return 1.0, truth
        return float(tested), float(generator.binomial(tested, truth))

    def key_of(samples: ArrayLike, tested: ArrayLike, errors: ArrayLike
               ) -> NDArray[np.float64]:
        """The bound the controller believes it would settle up with."""
        length = np.asarray(samples, dtype=float)
        published = np.asarray(tested, dtype=float)
        observed = np.divide(np.asarray(errors, dtype=float),
                             np.where(published > 0, published, 1.0))
        keys = bb84_finite_key_length(sifted_rate_hz * length * dt_s, observed,
                                      **finite_key_kwargs)
        return np.where((length > 0) & (published > 0), keys, 0.0)

    starts: list[int] = []
    ends: list[int] = []
    open_at: int | None = None
    block_tested = block_errors = 0.0
    link_tested = link_errors = 0.0

    def close(at: int) -> None:
        """Settle up, and only keep what the controller believes is worth it.

        A block whose own estimate says it will yield nothing is never
        processed by a receiver — the bits are dropped rather than run through
        privacy amplification. Keeping it here would be scoring the controller
        on a decision it would not make. The estimate is the controller's, so
        a block it wrongly writes off is a real loss and shows up as one.
        """
        nonlocal open_at, block_tested, block_errors
        if (open_at is not None and at > open_at
                and float(key_of(at - open_at, block_tested, block_errors)) > 0.0):
            starts.append(open_at)
            ends.append(at)
        open_at = None
        block_tested = block_errors = 0.0

    for index in range(1, edges.size):
        segment_start, segment_end = int(edges[index - 1]), int(edges[index])
        tested, errors = disclose(segment_start, segment_end)
        link_tested += tested
        link_errors += errors
        hopeless = tested > 0 and errors / tested > abort_qber

        if open_at is None:
            if hopeless:
                continue
            open_at, block_tested, block_errors = segment_start, 0.0, 0.0

        if segment_start > open_at:
            # What the rest of the horizon is worth if the link stays as it has
            # been measured: the best rate a stationary link of that quality
            # sustains, which is the one its whole budget in a single block —
            # the finite-size cost per bit falls monotonically with length.
            forecast = link_errors / link_tested if link_tested > 0 else 1.0
            horizon_key = float(key_of(maximum_samples,
                                       disclosed_per_sample * maximum_samples,
                                       disclosed_per_sample * maximum_samples
                                       * forecast))
            per_sample = horizon_key / maximum_samples

            horizon_end = open_at + maximum_samples
            banked = float(key_of(segment_start - open_at, block_tested,
                                  block_errors))
            closing = banked + per_sample * (horizon_end - segment_start)

            # Keeping it open: carry the block through this segment to the best
            # closing point the grid offers, filling what follows the segment
            # with fibre at the forecast quality.
            later = edges[(edges >= segment_end) & (edges <= horizon_end)]
            if later.size == 0:
                later = np.array([segment_end], dtype=np.int64)
            filled = (later - segment_end).astype(float)
            keeping = key_of(later - open_at,
                             block_tested + tested + disclosed_per_sample * filled,
                             block_errors + errors
                             + disclosed_per_sample * filled * forecast)
            keeping = keeping + per_sample * (horizon_end - later)

            # ``>=`` rather than ``>`` so that a tie closes. Ties happen in one
            # place only — both options worth nothing, which is a block that
            # has gone bad on a link whose future is also bad — and there the
            # right move is to drop it and start clean rather than keep
            # accumulating into a block that can no longer pay.
            if closing >= float(np.max(keeping)):
                close(segment_start)
                if hopeless:
                    continue
                open_at, block_tested, block_errors = segment_start, 0.0, 0.0

        block_tested += tested
        block_errors += errors
        if segment_end - open_at >= maximum_samples:
            close(segment_end)

    close(int(edges[-1]))

    if not starts:
        return BlockPlan(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0), dt_s, error.size, strategy="adaptativo")

    start_array = np.array(starts, dtype=np.int64)
    end_array = np.array(ends, dtype=np.int64)
    key_bits = _block_key_bits(prefix, start_array, end_array, sifted_rate_hz,
                               dt_s, **finite_key_kwargs)
    return BlockPlan(start_array, end_array, key_bits, dt_s, error.size,
                     strategy="adaptativo")
