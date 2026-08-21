"""
Small pilot on MATH/intermediate_algebra to get REAL effect-size and
per-sample variance numbers for the three components that have never
been tested on this suite (tool_use, planning, multi_critic -- only
cot/critique have any MATH history, and that data predates the
MATH_SYSTEM prompt-leak fix, so it's not clean either -- see
process_log.md's "MATH_SYSTEM prompt-leak fix" entry).

WHY: before committing multi-day compute to a full 16-combo x 5-seed x
n=500 x 2-model Shapley sweep on MATH, get real numbers to run a power
calculation against FIRST (power_analysis.py). GPQA's n=198/5-seeds was
validated once, against one specific big effect (bare vs cot, +7pts) --
then reused to test ~20 much smaller Shapley main-effect and interaction
terms it was never checked against. Every one of those died under
multiple-comparison correction (see process_log.md's "sample-level
paired significance test" entry). That's the actual root cause of the
failed sweep, not bad luck -- fix the process this time, not just the
suite.

DESIGN: bare + each of the 4 single-component configs (5 configs, not
the full 16 -- isolated ablation deltas are enough to estimate a
component's typical effect size/variance for a power calculation; the
full Shapley grid is only needed for the real sweep, once sizing is
known). n=40, 3 seeds, both locked-in models. Saves per-seed log paths
(not just accuracy) so power_analysis.py can pull real per-SAMPLE
outcomes, not just 3 noisy seed-means -- same lesson as GPQA's
seed-level-bootstrap-vs-sample-level-test gap.

Usage:
    python math_pilot.py
"""

import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

from elicit_task import elicit
from run_shapley_sweep import EXTRA_BODY, MAX_TOKENS, MODEL_PAIR, config_label, get_accuracy

SUITE = "math"
LIMIT = 40
TEMPERATURE = 0.7
SEEDS = [1, 2, 3]
MAX_CONNECTIONS = 5

TOGGLES = ["critique", "tool_use", "planning", "multi_critic"]
CONFIGS = [
    {t: (t == active) for t in TOGGLES}
    for active in [None] + TOGGLES
]


def _save_summary(summary_path: Path, model: str, critic_model: str, results: dict):
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": model,
                "critic_model": critic_model,
                "suite": SUITE,
                "math_subject": "intermediate_algebra",
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


def run_for_model(model: str, critic_model: str):
    model_slug = model.replace("/", "-").replace(":", "-").lower()
    Path("results").mkdir(exist_ok=True)
    summary_path = Path(f"results/math_pilot_{model_slug}.json")

    results: dict[str, list[tuple[int, float, str]]] = {}
    done: set[tuple[str, int]] = set()
    if summary_path.exists():
        with open(summary_path) as f:
            prior = json.load(f)
        for label, runs in prior.get("results", {}).items():
            results[label] = [(r["seed"], r["accuracy"], r["log"]) for r in runs]
            for r in runs:
                done.add((label, r["seed"]))
        if done:
            print(f"Resuming {summary_path}: {len(done)} (config, seed) runs "
                  f"already complete, skipping those.\n")

    print(f"\n=== MATH PILOT model={model}  critic_model={critic_model} "
          f"({len(CONFIGS)} configs x {len(SEEDS)} seeds, n={LIMIT}) ===\n")

    for cfg in CONFIGS:
        label = config_label(cfg)
        results.setdefault(label, [])
        for seed in SEEDS:
            if (label, seed) in done:
                print(f"{label:16s} seed={seed}  SKIPPED (already in {summary_path})")
                continue
            kwargs = dict(cfg)
            if kwargs.get("multi_critic"):
                kwargs["critic_model"] = critic_model
                # see run_shapley_sweep.py's identical comment: the critic
                # model needs its own config passed explicitly, or it falls
                # back to its raw server default and can hit the same
                # max_tokens overflow already fixed for the primary model.
                kwargs["critic_config"] = GenerateConfig(
                    max_tokens=MAX_TOKENS, extra_body=EXTRA_BODY,
                )
            log = inspect_eval(
                elicit(suite=SUITE, **kwargs),
                model=model,
                limit=LIMIT,
                temperature=TEMPERATURE,
                seed=seed,
                log_dir="./logs",
                max_connections=MAX_CONNECTIONS,
                max_tokens=MAX_TOKENS,
                extra_body=EXTRA_BODY,
            )[0]
            acc = get_accuracy(log)
            log_path = str(log.location) if hasattr(log, "location") else None
            results[label].append((seed, acc, log_path))
            print(f"{label:16s} seed={seed}  accuracy={acc:.3f}  log={log_path}")
            _save_summary(summary_path, model, critic_model, results)

    _save_summary(summary_path, model, critic_model, results)
    print(f"\nSaved -> {summary_path}")


def main():
    print(f"{len(CONFIGS)} configs x {len(SEEDS)} seeds x {len(MODEL_PAIR)} models "
          f"= {len(CONFIGS) * len(SEEDS) * len(MODEL_PAIR)} total runs, n={LIMIT} each\n")
    for model, critic_model in MODEL_PAIR:
        run_for_model(model, critic_model)
    print("\nBoth models done. Next: python power_analysis.py "
          "results/math_pilot_*.json")


if __name__ == "__main__":
    main()
