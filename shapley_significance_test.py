"""
Sample-level paired significance test for Shapley main effects and
pairwise interaction terms -- extends mcnemar_test.py's per-sample
pairing idea to the full Shapley/interaction linear combinations,
instead of just two configs at a time.

WHY THIS EXISTS: shapley_attribution.py's bootstrap CI resamples only
the 5 per-seed COALITION MEAN accuracies for each of the 16 coalitions
-- 5 numbers per coalition, full stop. But every coalition mean is
itself an average over 198 real per-sample outcomes, and -- exactly
like every paired comparison already done in this project via
mcnemar_test.py -- those per-sample outcomes are directly comparable
across coalitions, because every config runs on the IDENTICAL 198 GPQA
questions in the IDENTICAL order (shuffle=False). Treating each
coalition as 5 opaque numbers throws away almost all of that real
per-sample pairing power.

METHOD: every Shapley main effect and every pairwise interaction index
is a fixed weighted linear combination of coalition MEAN accuracies
(see shapley_attribution.py's shapley_values()/pairwise_interactions()
for the coefficients). Since each coalition mean is itself a per-sample
mean, the identical weighted combination can be computed PER (sample,
seed) pair instead of per coalition -- one real-valued "does this
sample/seed support a positive effect" observation per (k, s), instead
of one number per coalition. Averaging those over all (k, s) reproduces
the EXACT SAME point estimate as the coalition-mean version (checked
below as a hard assertion, not just eyeballed) -- but now there are
~990 raw observations (198 samples x 5 seeds) to run a real paired
test on, instead of a 5-point bootstrap.

TESTS: one-sample t-test (parametric, tests mean != 0) and Wilcoxon
signed-rank (nonparametric, doesn't assume the per-(sample,seed)
statistic is normally distributed -- it isn't; it's a weighted sum of
0/1 differences, so it's discrete-ish) against a null of 0, for every
main effect and every pairwise interaction.

CAVEAT, same one already flagged in mcnemar_test.py's pooled mode and
every pooled-McNemar entry in process_log.md: the 5 seeds share the
same fixed 198-question set, so treat this as an approximation to
independent draws, not textbook-exact.

Usage:
    python shapley_significance_test.py results/shapley_sweep_gpqa_together-qwen-qwen2.5-7b-instruct-turbo.json
"""

import itertools
import json
import math
import sys

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

from mcnemar_test import sample_outcomes
from run_shapley_sweep import TOGGLES, config_label
from shapley_attribution import pairwise_interactions, shapley_values

SEEDS = [1, 2, 3, 4, 5]


def load_all_outcomes(path: str) -> dict[frozenset, dict[int, dict[str, bool]]]:
    """coalition -> seed -> {sample_id: correct}. Reads every one of the
    16*5=80 per-seed eval logs referenced in the summary JSON."""
    with open(path) as f:
        data = json.load(f)

    label_to_components = {
        config_label({t: on for t, on in zip(TOGGLES, combo)}): frozenset(
            t for t, on in zip(TOGGLES, combo) if on
        )
        for combo in itertools.product([False, True], repeat=len(TOGGLES))
    }

    by_coalition: dict[frozenset, dict[int, dict[str, bool]]] = {}
    for label, runs in data["results"].items():
        coalition = label_to_components[label]
        by_coalition[coalition] = {}
        for run in runs:
            print(f"  reading {label} seed={run['seed']}...", file=sys.stderr)
            by_coalition[coalition][run["seed"]] = sample_outcomes(run["log"])

    return by_coalition


def per_sample_seed_matrix(
    outcomes: dict[frozenset, dict[int, dict[str, bool]]], coalition: frozenset
) -> dict[int, dict[str, float]]:
    """seed -> {sample_id: 1.0/0.0} for one coalition, as floats."""
    return {
        seed: {sid: (1.0 if correct else 0.0) for sid, correct in samples.items()}
        for seed, samples in outcomes[coalition].items()
    }


def shared_sample_ids(outcomes, coalitions_needed: set[frozenset]) -> dict[int, set[str]]:
    """seed -> sample ids present in EVERY needed coalition for that seed."""
    result = {}
    for seed in SEEDS:
        common = None
        for c in coalitions_needed:
            ids = set(outcomes[c][seed])
            common = ids if common is None else (common & ids)
        result[seed] = common
    return result


def main_effect_terms(player: str) -> list[tuple[frozenset, frozenset, float]]:
    """(plus_coalition, minus_coalition, weight) triples for one Shapley main effect."""
    n = len(TOGGLES)
    others = [t for t in TOGGLES if t != player]
    terms = []
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            s = frozenset(combo)
            weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
            terms.append((s | {player}, s, weight))
    return terms


def interaction_terms(i: str, j: str) -> list[tuple[frozenset, frozenset, frozenset, frozenset, float]]:
    """(plus_ij, plus_i, plus_j, minus, weight) quadruples for one pairwise interaction."""
    n = len(TOGGLES)
    others = [t for t in TOGGLES if t not in (i, j)]
    terms = []
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            s = frozenset(combo)
            weight = math.factorial(len(s)) * math.factorial(n - len(s) - 2) / math.factorial(n - 1)
            terms.append((s | {i, j}, s | {i}, s | {j}, s, weight))
    return terms


def per_sample_main_effect(outcomes, player: str) -> np.ndarray:
    terms = main_effect_terms(player)
    needed = {c for term in terms for c in term[:2]}
    shared = shared_sample_ids(outcomes, needed)

    values = []
    for seed in SEEDS:
        ids = sorted(shared[seed])
        for sid in ids:
            d = 0.0
            for plus, minus, weight in terms:
                x_plus = outcomes[plus][seed][sid]
                x_minus = outcomes[minus][seed][sid]
                d += weight * ((1.0 if x_plus else 0.0) - (1.0 if x_minus else 0.0))
            values.append(d)
    return np.array(values)


def per_sample_interaction(outcomes, i: str, j: str) -> np.ndarray:
    terms = interaction_terms(i, j)
    needed = {c for term in terms for c in term[:4]}
    shared = shared_sample_ids(outcomes, needed)

    values = []
    for seed in SEEDS:
        ids = sorted(shared[seed])
        for sid in ids:
            d = 0.0
            for plus_ij, plus_i, plus_j, minus, weight in terms:
                x_ij = 1.0 if outcomes[plus_ij][seed][sid] else 0.0
                x_i = 1.0 if outcomes[plus_i][seed][sid] else 0.0
                x_j = 1.0 if outcomes[plus_j][seed][sid] else 0.0
                x_0 = 1.0 if outcomes[minus][seed][sid] else 0.0
                d += weight * (x_ij - x_i - x_j + x_0)
            values.append(d)
    return np.array(values)


def report_test(label: str, values: np.ndarray, coalition_mean_estimate: float):
    n = len(values)
    point = float(np.mean(values))
    # Sanity check: sample-level mean must reproduce the coalition-mean
    # Shapley/interaction value exactly (same linear combination,
    # computed a different way) -- if this drifts, something's wrong
    # with the term-generation logic above, not with the data.
    assert abs(point - coalition_mean_estimate) < 1e-9, (
        f"{label}: sample-level point estimate {point} != "
        f"coalition-mean estimate {coalition_mean_estimate}"
    )

    t_p = ttest_1samp(values, popmean=0.0).pvalue
    try:
        w_p = wilcoxon(values).pvalue
    except ValueError:
        w_p = float("nan")  # all-zero differences, degenerate

    flag = "  <- SIGNIFICANT (both tests p<0.05)" if (t_p < 0.05 and w_p < 0.05) else \
           "  <- significant (t-test only)" if t_p < 0.05 else \
           "  <- significant (Wilcoxon only)" if w_p < 0.05 else ""
    print(f"  {label:24s} {point:+.4f}  n={n:4d}  t-test p={t_p:.4f}  "
          f"wilcoxon p={w_p:.4f}{flag}")


def analyze(path: str):
    with open(path) as f:
        data = json.load(f)
    model_label = data["model"]

    print(f"\n{'=' * 72}")
    print(f"{model_label}  ({path})")
    print(f"{'=' * 72}")

    from shapley_attribution import load_seed_accuracies
    by_coalition_acc = load_seed_accuracies(path)
    v_mean = {c: float(np.mean(accs)) for c, accs in by_coalition_acc.items()}
    coalition_sv = shapley_values(v_mean)
    coalition_iv = pairwise_interactions(v_mean)

    print("\nReading all 80 per-seed eval logs (this takes a minute)...", file=sys.stderr)
    outcomes = load_all_outcomes(path)

    print("\nMain effects (sample-level paired test, n up to 198*5=990):")
    for player in TOGGLES:
        values = per_sample_main_effect(outcomes, player)
        report_test(player, values, coalition_sv[player])

    print("\nPairwise interactions (sample-level paired test):")
    for i, j in itertools.combinations(TOGGLES, 2):
        values = per_sample_interaction(outcomes, i, j)
        report_test(f"{i} x {j}", values, coalition_iv[(i, j)])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze(path)


if __name__ == "__main__":
    main()
