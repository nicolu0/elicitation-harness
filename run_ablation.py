"""
Run every on/off combination of the toggled components, across multiple
SEEDS at a non-zero TEMPERATURE, and print mean accuracy per config.

WHY TEMPERATURE > 0 AND MULTIPLE SEEDS:
At temperature 0 the model is (near-)deterministic, so re-running the same
198 GPQA questions N times just reproduces the same answers -- no new
information, and McNemar's test stays bottlenecked on however many
questions the two configs happened to disagree on in that single run.
Raising temperature introduces real sampling variation, and running
multiple seeds gives McNemar's test more discordant-pair observations to
pool across, which is what actually buys more statistical power when the
underlying dataset size (e.g. GPQA Diamond's 198 questions) is fixed and
can't just be scaled up with a bigger --limit.

TRADE-OFF, stated explicitly (put this in your methodology section too):
Temperature 0 measures "the model's single best (greedy) answer."
Temperature > 0 measures "the model's expected accuracy under stochastic
sampling" -- a related but genuinely different quantity, and repeated
seeds on the SAME 198 questions are not fully independent draws, so any
resulting p-value should be read as an approximation, not textbook-exact.
Keep temperature IDENTICAL across every config you compare -- comparing a
temp=0 config against a temp=0.7 config would confound the scaffold effect
with a temperature effect, exactly the kind of confound this project is
built to avoid introducing elsewhere.

Usage:
    python run_ablation.py

Output:
    - printed per-config mean accuracy (+ per-seed spread)
    - results/ablation_summary.json -- log paths grouped by (config, seed),
      consumed by mcnemar_test.py's pooled mode to compute paired
      significance across all seeds at once, not just one run.

Next steps (project step 5):
  - add more components (tool use, best-of-N, multi-critic)
  - compute Shapley attribution over all 2^n configs
  - plot capability vs cost (Pareto frontier)
"""

import itertools
import json
import statistics
from pathlib import Path

from inspect_ai import eval as inspect_eval

from elicit_task import elicit

# Swap this for your open-model pair once the pipe works, e.g.
#   "together/Qwen/Qwen2.5-7B-Instruct-Turbo"   (CONFIRM the current id!)
MODEL = "openai/gpt-4o-mini"

# Which registered adapter to run (see ADAPTERS in elicit_task.py):
# "gsm8k" | "math" | "gpqa"
SUITE = "gpqa"

# Samples per config per seed. For GPQA Diamond this is capped at 198
# (the whole dataset) -- raising LIMIT past that does nothing.
LIMIT = 200

# Sampling temperature, applied IDENTICALLY to every config below.
# 0.0 = deterministic (use for a quick single-run smoke test).
# ~0.7 = real variation across seeds, still coherent reasoning quality.
TEMPERATURE = 0.7

# How many independent runs per config. More seeds = more discordant-pair
# observations for McNemar pooling = more statistical power, at linearly
# increasing cost. 5 is a reasonable starting point.
SEEDS = [1, 2, 3, 4, 5]

# All four combinations of (cot, critique). Add keys here as you add components.
CONFIGS = [
    {"cot": cot, "critique": cr}
    for cot, cr in itertools.product([False, True], [False, True])
]


def config_label(cfg: dict) -> str:
    return "+".join(k for k, v in cfg.items() if v) or "bare"


def get_accuracy(log) -> float:
    """Pull the accuracy metric out of an EvalLog defensively."""
    try:
        for score in log.results.scores:
            metric = score.metrics.get("accuracy")
            if metric is not None:
                return metric.value
    except Exception:
        pass
    return float("nan")


def main():
    print(
        f"model={MODEL}  suite={SUITE}  limit={LIMIT}  "
        f"temperature={TEMPERATURE}  seeds={SEEDS}\n"
    )

    # {config_label: [(seed, accuracy, log_path), ...]}
    results: dict[str, list[tuple[int, float, str]]] = {}

    for cfg in CONFIGS:
        label = config_label(cfg)
        results[label] = []
        for seed in SEEDS:
            log = inspect_eval(
                elicit(suite=SUITE, **cfg),
                model=MODEL,
                limit=LIMIT,
                temperature=TEMPERATURE,
                seed=seed,
                log_dir="./logs",
            )[0]
            acc = get_accuracy(log)
            log_path = str(log.location) if hasattr(log, "location") else None
            results[label].append((seed, acc, log_path))
            print(f"{label:16s} seed={seed}  accuracy={acc:.3f}  log={log_path}")

    print("\n=== elicitation slice (mean over seeds) ===")
    means = {}
    for label, runs in results.items():
        accs = [a for _, a, _ in runs]
        mean_acc = statistics.mean(accs)
        spread = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        means[label] = mean_acc
        print(f"{label:16s} mean={mean_acc:.3f}  (seed stdev={spread:.3f}, n_seeds={len(accs)})")

    bare_mean = means.get("bare", float("nan"))
    print("\n=== deltas vs bare (mean) ===")
    for label, mean_acc in means.items():
        delta = "" if label == "bare" else f"   (delta vs bare: {mean_acc - bare_mean:+.3f})"
        print(f"{label:16s} mean={mean_acc:.3f}{delta}")

    # Save log paths grouped by config so mcnemar_test.py can pool across
    # seeds for a proper paired significance test, not just eyeball means.
    Path("results").mkdir(exist_ok=True)
    summary_path = Path("results/ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "suite": SUITE,
                "limit": LIMIT,
                "temperature": TEMPERATURE,
                "seeds": SEEDS,
                "results": {
                    label: [
                        {"seed": s, "accuracy": a, "log": p} for s, a, p in runs
                    ]
                    for label, runs in results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved log paths -> {summary_path}")
    print(
        "Run `python mcnemar_test.py --pooled results/ablation_summary.json "
        "bare cot` to test bare vs cot pooled across all seeds."
    )
    print("Run `inspect view` to inspect transcripts and token/cost usage.")


if __name__ == "__main__":
    main()
