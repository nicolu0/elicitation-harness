"""
Shapley attribution + pairwise interaction terms over the 4-component
sweep (run_shapley_sweep.py). This is the analysis step referenced in
that script's own closing print -- "Shapley attribution + interaction
term computation ... not yet built."

WHY SHAPLEY, NOT JUST READING OFF THE ABLATION DELTAS: the raw
bare-vs-single-component deltas used earlier in this project (see
writeup/process_log.md's Phase 4 entries) only tell you a component's
effect in ONE context (added to nothing else). But whether e.g. critique
helps depends on what else is already on -- that's exactly the
cot+critique-is-worse-than-cot-alone pattern the process log already
found by hand, repeatedly, on both GPQA and MATH. Shapley values fix
this by averaging a component's marginal contribution over EVERY
possible context (every subset of the other components), weighted so
each "coalition size" contributes equally overall. With exactly
2^4=16 measured coalitions (this sweep measures every subset, not a
sample of orderings), the Shapley value here is EXACT, not estimated.

INTERACTION TERMS: the Shapley INTERACTION index (Grabisch & Roubens
1999) extends the same idea to pairs -- it answers "does turning on
components i and j TOGETHER beat what you'd predict from their two
separate main effects," again averaged over every context the other
two components could be in. A positive interaction means the pair
synergizes; negative means they step on each other (the
critique+cot pattern, quantified rather than eyeballed).

UNCERTAINTY: each of the 16 coalitions was measured with 5 seeds, not
1 -- that's real, usable variance, not just a point estimate. Bootstrap
over seeds (resample each coalition's 5 seeds with replacement,
recompute every Shapley value and interaction term, repeat 5000x) to
get a 95% percentile CI on every number reported, rather than reporting
a single number with no sense of whether it's noise.

Usage:
    python shapley_attribution.py results/shapley_sweep_gpqa_together-qwen-qwen2.5-7b-instruct-turbo.json
    python shapley_attribution.py results/shapley_sweep_gpqa_together-qwen-qwen2.5-7b-instruct-turbo.json results/shapley_sweep_gpqa_together-google-gemma-3n-e4b-it.json
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

from run_shapley_sweep import TOGGLES, config_label

N_BOOTSTRAP = 5000
RNG_SEED = 0


def load_seed_accuracies(path: str) -> dict[frozenset, list[float]]:
    """label (e.g. "critique+planning") -> list of per-seed accuracies,
    keyed by the frozenset of active components so it's independent of
    label string ordering."""
    with open(path) as f:
        data = json.load(f)

    label_to_components = {
        config_label({t: on for t, on in zip(TOGGLES, combo)}): frozenset(
            t for t, on in zip(TOGGLES, combo) if on
        )
        for combo in itertools.product([False, True], repeat=len(TOGGLES))
    }

    by_coalition: dict[frozenset, list[float]] = {}
    for label, runs in data["results"].items():
        if label not in label_to_components:
            raise ValueError(f"Unrecognized config label '{label}' in {path} "
                              f"-- does TOGGLES still match run_shapley_sweep.py?")
        coalition = label_to_components[label]
        accs = [r["accuracy"] for r in runs]
        if any(a != a for a in accs):  # nan check
            raise ValueError(
                f"{path}: '{label}' has a NaN accuracy -- rerun that seed "
                f"before computing attribution (see process_log.md's "
                f"'one more bad cell' entry for how this was handled before)."
            )
        if len(accs) != 5:
            raise ValueError(
                f"{path}: '{label}' has {len(accs)} seeds, expected 5 -- "
                f"sweep isn't complete yet."
            )
        by_coalition[coalition] = accs

    missing = set(label_to_components.values()) - set(by_coalition)
    if missing:
        raise ValueError(f"{path}: missing coalitions {missing} -- sweep isn't complete.")

    return by_coalition


def shapley_values(v: dict[frozenset, float]) -> dict[str, float]:
    """Exact Shapley value per component, given the full 2^4 value function."""
    n = len(TOGGLES)
    result = {}
    for player in TOGGLES:
        others = [t for t in TOGGLES if t != player]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
                marginal = v[s | {player}] - v[s]
                total += weight * marginal
        result[player] = total
    return result


def pairwise_interactions(v: dict[frozenset, float]) -> dict[tuple[str, str], float]:
    """Exact Shapley interaction index (Grabisch & Roubens) for every pair."""
    n = len(TOGGLES)
    result = {}
    for i, j in itertools.combinations(TOGGLES, 2):
        others = [t for t in TOGGLES if t not in (i, j)]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                weight = math.factorial(len(s)) * math.factorial(n - len(s) - 2) / math.factorial(n - 1)
                delta = v[s | {i, j}] - v[s | {i}] - v[s | {j}] + v[s]
                total += weight * delta
        result[(i, j)] = total
    return result


def bootstrap_ci(by_coalition: dict[frozenset, list[float]], rng: np.random.Generator):
    """Resample each coalition's 5 seeds with replacement, N_BOOTSTRAP times,
    recomputing Shapley values + pairwise interactions each time. Returns
    (shapley_samples, interaction_samples) as dicts of lists."""
    shapley_samples = {t: [] for t in TOGGLES}
    interaction_samples = {pair: [] for pair in itertools.combinations(TOGGLES, 2)}

    coalitions = list(by_coalition)
    for _ in range(N_BOOTSTRAP):
        v_resampled = {}
        for coalition in coalitions:
            accs = by_coalition[coalition]
            resampled = rng.choice(accs, size=len(accs), replace=True)
            v_resampled[coalition] = float(np.mean(resampled))

        sv = shapley_values(v_resampled)
        for t, val in sv.items():
            shapley_samples[t].append(val)

        iv = pairwise_interactions(v_resampled)
        for pair, val in iv.items():
            interaction_samples[pair].append(val)

    return shapley_samples, interaction_samples


def ci_str(samples: list[float]) -> str:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return f"[{lo:+.4f}, {hi:+.4f}]"


def analyze_one(path: str, model_label: str):
    by_coalition = load_seed_accuracies(path)
    v_mean = {coalition: float(np.mean(accs)) for coalition, accs in by_coalition.items()}

    print(f"\n{'=' * 72}")
    print(f"{model_label}  ({path})")
    print(f"{'=' * 72}")

    print("\nCoalition means (all 16, sorted by accuracy):")
    for coalition, acc in sorted(v_mean.items(), key=lambda kv: -kv[1]):
        label = "+".join(sorted(coalition)) or "bare"
        stdev = float(np.std(by_coalition[coalition], ddof=1))
        print(f"  {label:40s} {acc:.4f}  (seed stdev {stdev:.4f})")

    sv = shapley_values(v_mean)
    iv = pairwise_interactions(v_mean)

    # Efficiency check: Shapley values must sum to v(full) - v(empty).
    full, empty = frozenset(TOGGLES), frozenset()
    expected_sum = v_mean[full] - v_mean[empty]
    actual_sum = sum(sv.values())
    assert abs(expected_sum - actual_sum) < 1e-9, (
        f"Shapley efficiency check failed: {actual_sum} != {expected_sum}"
    )

    print(f"\nSanity check (efficiency property): sum of Shapley values "
          f"= {actual_sum:+.4f}, v(full)-v(bare) = {expected_sum:+.4f} -- match.")

    print("\nBootstrapping 95% CIs over seed resampling "
          f"({N_BOOTSTRAP} iterations)...")
    rng = np.random.default_rng(RNG_SEED)
    shapley_samples, interaction_samples = bootstrap_ci(by_coalition, rng)

    print("\nShapley main effects (mean marginal contribution, all contexts):")
    for t in sorted(TOGGLES, key=lambda t: -sv[t]):
        significant = not (np.percentile(shapley_samples[t], 2.5) < 0 < np.percentile(shapley_samples[t], 97.5))
        flag = "  <- CI excludes 0" if significant else ""
        print(f"  {t:16s} {sv[t]:+.4f}  95% CI {ci_str(shapley_samples[t])}{flag}")

    print("\nPairwise Shapley interactions (synergy > 0, redundancy/conflict < 0):")
    for pair in sorted(iv, key=lambda p: -abs(iv[p])):
        i, j = pair
        significant = not (np.percentile(interaction_samples[pair], 2.5) < 0 < np.percentile(interaction_samples[pair], 97.5))
        flag = "  <- CI excludes 0" if significant else ""
        print(f"  {i} x {j:16s} {iv[pair]:+.4f}  95% CI {ci_str(interaction_samples[pair])}{flag}")

    return v_mean, sv, iv


def _short_model_name(label: str) -> str:
    """'together/Qwen/Qwen2.5-7B-Instruct-Turbo' -> 'Qwen2.5-7B-Instruct-Turbo'"""
    return label.rsplit("/", 1)[-1]


def compare_models(results: dict[str, tuple]):
    labels = list(results)
    if len(labels) < 2:
        return
    short = {lbl: _short_model_name(lbl) for lbl in labels}
    col_width = max(18, max(len(s) for s in short.values()) + 2)

    print(f"\n{'=' * 72}")
    print("CROSS-MODEL COMPARISON (Shapley main effects)")
    print(f"{'=' * 72}")
    header = f"  {'component':16s}" + "".join(f"{short[lbl]:>{col_width}s}" for lbl in labels)
    print(header)
    for t in TOGGLES:
        row = f"  {t:16s}"
        for lbl in labels:
            _, sv, _ = results[lbl]
            row += f"{sv[t]:+{col_width}.4f}"
        print(row)

    print(f"\n{'=' * 72}")
    print("CROSS-MODEL COMPARISON (pairwise interactions)")
    print(f"{'=' * 72}")
    header = f"  {'pair':20s}" + "".join(f"{short[lbl]:>{col_width}s}" for lbl in labels)
    print(header)
    for pair in itertools.combinations(TOGGLES, 2):
        i, j = pair
        pair_label = f"{i}+{j}"
        row = f"  {pair_label:20s}"
        for lbl in labels:
            _, _, iv = results[lbl]
            row += f"{iv[pair]:+{col_width}.4f}"
        print(row)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:]
    results = {}
    for path in paths:
        with open(path) as f:
            model_label = json.load(f)["model"]
        results[model_label] = analyze_one(path, model_label)

    compare_models(results)


if __name__ == "__main__":
    main()
