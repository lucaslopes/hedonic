"""LFK overlapping NMI against a dense, independent transcription of its definition.

Lancichinetti, Fortunato & Kertész (2009), Appendix B: a pair of communities X, Y
counts only when h(p11) + h(p00) > h(p01) + h(p10), with h(p) = -p log2 p; then
H(X|Y) = H(X, Y) - H(Y), otherwise H(X|Y) = H(X). Each community keeps its best
(smallest) conditional entropy, normalized by its own entropy, and
ONMI = 1 - (mean over X + mean over Y) / 2. The reference below scores every pair
(zero-overlap pairs included) with these formulas and no code from hedonic.metrics.
"""

from __future__ import annotations

import math
import random

from hedonic.metrics import overlapping_normalized_mutual_information_lfk as onmi


def _h(p: float) -> float:
    return -p * math.log2(p) if p > 0.0 else 0.0


def _binary(p: float) -> float:
    return _h(p) + _h(1.0 - p)


def _reference(predicted, ground_truth, n_vertices=None) -> float:
    xs = sorted({frozenset(c) for c in predicted if c}, key=sorted)
    ys = sorted({frozenset(c) for c in ground_truth if c}, key=sorted)
    if not xs and not ys:
        return 1.0
    if not xs or not ys:
        return 0.0
    if set(xs) == set(ys):
        return 1.0
    n = n_vertices if n_vertices is not None else len(set().union(*xs, *ys))

    def side(sources, targets) -> float:
        total = 0.0
        for x in sources:
            h_x = _binary(len(x) / n)
            best = h_x
            for y in targets:
                both = len(x & y)
                p11, p10 = both / n, (len(x) - both) / n
                p01, p00 = (len(y) - both) / n, (n - len(x) - len(y) + both) / n
                if _h(p11) + _h(p00) > _h(p01) + _h(p10):
                    joint = _h(p11) + _h(p10) + _h(p01) + _h(p00)
                    best = min(best, max(0.0, joint - _binary(len(y) / n)))
            total += 1.0 if h_x == 0.0 else best / h_x
        return total / len(sources)

    return max(0.0, min(1.0, 1.0 - 0.5 * (side(xs, ys) + side(ys, xs))))


def test_barely_overlapping_small_pair_gets_no_credit():
    # Two 10-vertex communities sharing 3 vertices in a universe of 1000:
    # h(p11) + h(p00) = 0.049 < h(p01) + h(p10) = 0.100, so neither conditions on
    # the other and ONMI is 0 (binary entropies in the test would give 0.147).
    predicted = [list(range(10))]
    ground_truth = [[0, 1, 2] + list(range(10, 17))]
    assert onmi(predicted, ground_truth, n_vertices=1000) == 0.0
    assert _reference(predicted, ground_truth, n_vertices=1000) == 0.0


def test_matches_dense_reference_on_random_covers():
    rng = random.Random(20261112)
    for case in range(300):
        n = rng.randint(8, 60)
        vertices = list(range(n))

        def cover():
            return [rng.sample(vertices, rng.randint(1, max(1, n // 3))) for _ in range(rng.randint(1, 6))]

        predicted, ground_truth = cover(), cover()
        for universe in (None, n + rng.randint(0, 3 * n)):
            got = onmi(predicted, ground_truth, n_vertices=universe)
            want = _reference(predicted, ground_truth, n_vertices=universe)
            assert abs(got - want) <= 1e-12, (case, universe, got, want)


def test_identical_and_disjoint_extremes():
    cover = [[0, 1, 2], [2, 3, 4, 5]]
    assert onmi(cover, [list(c) for c in reversed(cover)]) == 1.0
    assert onmi([[0, 1, 2]], [[3, 4, 5]], n_vertices=100) == 0.0
