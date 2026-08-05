"""
Run every on/off combination of the toggled components on a small slice and
print accuracy per config. The gap between `bare` and the best config is your
first measured elicitation gap.

Usage:
    python run_ablation.py

This is the seed of the real sweep runner. Next steps (project step 5):
  - add more components (tool use, best-of-N, multi-critic)
  - run multiple seeds and report mean +/- bootstrap CI
  - compute Shapley attribution over all 2^n configs
  - plot capability vs cost (Pareto frontier)
"""

import itertools

from inspect_ai import eval as inspect_eval

from elicit_task import elicit

# Swap this for your open-model pair once the pipe works, e.g.
#   "together/Qwen/Qwen2.5-7B-Instruct-Turbo"   (CONFIRM the current id!)
MODEL = "openai/gpt-4o-mini"

# Which registered adapter to run (see ADAPTERS in elicit_task.py):
# "gsm8k" | "math" | "gpqa"
SUITE = "gpqa"

# Samples per config. 200 is the real-sweep size; drop back to ~20-30 for a
# quick smoke test after changing prompts/scorers.
LIMIT = 200

# All four combinations of (cot, critique). Add keys here as you add components.
CONFIGS = [
    {"cot": cot, "critique": cr}
    for cot, cr in itertools.product([False, True], [False, True])
]


def get_accuracy(log):
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
    print(f"model={MODEL}  suite={SUITE}  limit={LIMIT}\n")
    rows = []
    for cfg in CONFIGS:
        log = inspect_eval(
            elicit(suite=SUITE, **cfg),
            model=MODEL,
            limit=LIMIT,
            log_dir="./logs",
        )[0]
        label = "+".join(k for k, v in cfg.items() if v) or "bare"
        rows.append((label, get_accuracy(log)))

    bare = dict(rows).get("bare", float("nan"))
    print("\n=== elicitation slice ===")
    for label, acc in rows:
        delta = "" if label == "bare" else f"   (delta vs bare: {acc - bare:+.3f})"
        print(f"{label:16s} accuracy={acc:.3f}{delta}")
    print("\nRun `inspect view` to inspect transcripts and token/cost usage.")


if __name__ == "__main__":
    main()
