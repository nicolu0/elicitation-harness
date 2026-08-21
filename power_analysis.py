"""
Power calculation from math_pilot.py's small-n data: given real observed
effect sizes and per-sample variance for each single MATH component
(tool_use, planning, multi_critic -- the ones with no prior MATH data --
plus critique for reference), compute how large a sample the REAL
16-combo Shapley sweep would need to reliably detect effects of that
size, at a significance threshold corrected for the number of hypothesis
tests actually planned.

WHY: GPQA's n=198/5-seeds was chosen once, to detect ONE specific large
effect (bare vs cot, +7pts, see process_log.md's "Scope decision" entry)
-- then reused to test ~20 much smaller Shapley main-effect and
interaction terms it was never power-checked against. Every one of them
died under multiple-comparison correction (see "sample-level paired
significance test" entry). This script exists so the MATH sweep's
sample size is chosen against the effects it will ACTUALLY be tested
against, before spending multi-day compute on it.

METHOD: for each single-component config, pull real per-sample paired
outcomes against bare from the actual .eval logs (same technique as
mcnemar_test.py's sample_outcomes(), pooled across the pilot's seeds) --
not just the seed-mean accuracies, for the same reason
shapley_significance_test.py used sample-level data over seed-level
bootstrapping. From the observed mean delta and its standard deviation,
compute the paired-observation count N needed for the target power at a
Bonferroni-corrected alpha (default: correcting for 20 tests, matching
the full main-effects + pairwise-interactions design already run on
GPQA -- override with --num-tests if the real MATH sweep's design ends
up different).

IMPORTANT LIMITATION, stated explicitly rather than left implicit: this
sizes the LARGEST, most optimistic version of each effect -- an isolated
component-vs-bare delta. The real Shapley main effect averages that same
component's marginal contribution across every possible context (see
shapley_attribution.py), and GPQA's own data showed that marginal
contribution can vary a lot by context (e.g. multi_critic's per-context
delta on gemma ranged from -0.51pts to +3.13pts depending what else was
on) -- so the true Shapley effect is often SMALLER than the isolated
delta this pilot measures. Treat every N below as a lower bound, not a
guarantee. Pairwise interaction terms are not sized at all here (they're
differences of differences, with roughly double the variance of a main
effect for the same nominal size) -- expect interactions to need
noticeably more N than whatever a same-sized main effect requires above.

Usage:
    python power_analysis.py results/math_pilot_together-qwen-qwen2.5-7b-instruct-turbo.json
    python power_analysis.py results/math_pilot_*.json --num-tests 20 --power 0.8
"""

import argparse
import glob
import json

import numpy as np
from scipy.stats import norm

from mcnemar_test import sample_outcomes


def paired_deltas(bare_runs, treatment_runs) -> np.ndarray:
    """Per-(sample, seed) treatment-minus-bare outcome, pooled across
    every seed shared by both configs."""
    bare_by_seed = {r["seed"]: r["log"] for r in bare_runs}
    treat_by_seed = {r["seed"]: r["log"] for r in treatment_runs}
    shared_seeds = sorted(set(bare_by_seed) & set(treat_by_seed))

    deltas = []
    for seed in shared_seeds:
        bare_outcomes = sample_outcomes(bare_by_seed[seed])
        treat_outcomes = sample_outcomes(treat_by_seed[seed])
        shared_ids = set(bare_outcomes) & set(treat_outcomes)
        for sid in shared_ids:
            deltas.append(
                (1.0 if treat_outcomes[sid] else 0.0)
                - (1.0 if bare_outcomes[sid] else 0.0)
            )
    return np.array(deltas)


def required_n(delta: float, sigma: float, alpha: float, power: float) -> float:
    """Paired one-sample t-test sample size (normal approximation), for a
    two-sided test at significance alpha and the given power."""
    if delta == 0 or sigma == 0:
        return float("inf")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return ((z_alpha + z_power) * sigma / delta) ** 2


def analyze(path: str, alpha: float, power: float):
    with open(path) as f:
        data = json.load(f)
    model = data["model"]
    results = data["results"]

    if "bare" not in results:
        print(f"{path}: no 'bare' config found, skipping.")
        return

    bare_accs = [r["accuracy"] for r in results["bare"]]
    print(f"\n{'=' * 72}")
    print(f"{model}  ({path})")
    print(f"{'=' * 72}")
    print(f"bare accuracy (pilot, n={data['limit']}): "
          f"mean={np.mean(bare_accs):.3f}, seeds={[round(a, 3) for a in bare_accs]}")

    for label, runs in results.items():
        if label == "bare":
            continue
        deltas = paired_deltas(results["bare"], runs)
        if len(deltas) == 0:
            print(f"\n  {label}: no shared samples with bare, skipping.")
            continue
        mean_delta = float(np.mean(deltas))
        sigma = float(np.std(deltas, ddof=1))
        n_req = required_n(mean_delta, sigma, alpha, power)

        print(f"\n  {label}:")
        print(f"    observed effect (pilot, {len(deltas)} paired obs): "
              f"{mean_delta:+.4f}  (sigma={sigma:.4f})")
        if n_req == float("inf") or n_req > 1_000_000:
            print(f"    required N for {power:.0%} power at alpha={alpha:.5f}: "
                  f"effect too small/noisy to size meaningfully from this "
                  f"pilot -- would need an impractically large sample even "
                  f"taking the observed effect at face value.")
        else:
            print(f"    required N (paired sample x seed observations) for "
                  f"{power:.0%} power at alpha={alpha:.5f}: {n_req:.0f}")
            print(f"      e.g. 5 seeds -> n={n_req / 5:.0f} samples/seed  |  "
                  f"3 seeds -> n={n_req / 3:.0f} samples/seed  |  "
                  f"n=500 -> {n_req / 500:.1f} seeds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--num-tests", type=int, default=20,
                         help="Number of hypothesis tests the real sweep will "
                              "run (for Bonferroni correction). Default 20 "
                              "matches the full 4-main-effect + 6-interaction "
                              "x 2-model design already used on GPQA.")
    parser.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args()

    alpha = 0.05 / args.num_tests
    print(f"Using Bonferroni-corrected alpha = 0.05 / {args.num_tests} = {alpha:.5f}, "
          f"target power = {args.power:.0%}")
    print("NOTE: these N's size isolated component-vs-bare deltas (the most "
          "optimistic case). Real Shapley main effects average over more "
          "contexts and are often smaller; pairwise interactions need "
          "noticeably more N still. Treat every number below as a floor.")

    paths = []
    for p in args.paths:
        paths.extend(glob.glob(p))

    for path in paths:
        analyze(path, alpha, args.power)


if __name__ == "__main__":
    main()
