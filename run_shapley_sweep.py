"""
The 4-component Shapley sweep: all 2^4=16 on/off combinations of
{critique, tool_use, planning, multi_critic}, across both locked-in
models, on GPQA. This is a SEPARATE script from run_ablation.py
deliberately -- that script is/was running the live Phase 4 gpt-4o-mini
MATH-vs-GPQA comparison, and editing its MODEL/SUITE/CONFIGS mid-flight
would risk corrupting its checkpoint-resume. This script reuses the same
proven patterns (checkpoint after every seed, resume from a partial
run, capped concurrency) rather than rediscovering them.

SCOPE, per writeup/process_log.md's "Scope decision" and "Second model
selection" entries:
  - 4 of the README's 6 components (retrieval and best_of_n excluded --
    see phases.md's scope note for why)
  - 2 models: Qwen2.5-7B-Instruct-Turbo + google/gemma-3n-E4B-it
    (NOT Llama -- inaccessible on this Together account, see process_log)
  - GPQA only (no budget axis, no second suite, for now)
  - 5 seeds (kept, not cut to 3 -- see process_log's real power-analysis
    illustration: the existing bare-vs-cot GPQA finding would likely not
    have been significant at 3 seeds)

CRITIC MODEL: multi_critic's critic_model is set to "the other model in
the pair" for each run, so it's a genuine cross-model check, not an
accidental duplicate of `critique` (see multi_critic()'s docstring in
elicit_task.py for why critic_model=None would just be same-model
critique again).

COST: real per-component pilot data (n=25, GPQA, both models) puts the
full sweep at roughly $8.31 (additive approximation, likely a floor --
see process_log.md's pilot entry). Two real bugs were found and fixed
during that pilot before trusting these numbers: `tool_use` crashed
without a sandbox (fixed in elicit_task.py), and `planning` was silently
broken (missing a follow-up instruction, also fixed). Both fixes are
already in elicit_task.py by the time this script runs.

Usage:
    python run_shapley_sweep.py
"""

import itertools
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval

from elicit_task import elicit

SUITE = "gpqa"
LIMIT = 198  # GPQA Diamond's full set -- hard-capped, raising this does nothing
TEMPERATURE = 0.7
SEEDS = [1, 2, 3, 4, 5]
MAX_CONNECTIONS = 5  # see run_ablation.py's own comment on why this is capped

# (model string, critic model string for THIS model's multi_critic runs)
MODEL_PAIR = [
    ("together/Qwen/Qwen2.5-7B-Instruct-Turbo", "together/google/gemma-3n-E4B-it"),
    ("together/google/gemma-3n-E4B-it", "together/Qwen/Qwen2.5-7B-Instruct-Turbo"),
]

TOGGLES = ["critique", "tool_use", "planning", "multi_critic"]
CONFIGS = [
    {t: on for t, on in zip(TOGGLES, combo)}
    for combo in itertools.product([False, True], repeat=len(TOGGLES))
]


def config_label(cfg: dict) -> str:
    return "+".join(k for k, v in cfg.items() if v) or "bare"


def get_accuracy(log) -> float:
    try:
        for score in log.results.scores:
            metric = score.metrics.get("accuracy")
            if metric is not None:
                return metric.value
    except Exception:
        pass
    return float("nan")


def _save_summary(summary_path: Path, model: str, critic_model: str, results: dict):
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": model,
                "critic_model": critic_model,
                "suite": SUITE,
                "limit": LIMIT,
                "temperature": TEMPERATURE,
                "seeds": SEEDS,
                "components": TOGGLES,
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
    summary_path = Path(f"results/shapley_sweep_{SUITE}_{model_slug}.json")

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

    print(f"\n=== model={model}  critic_model={critic_model} "
          f"({len(CONFIGS)} configs x {len(SEEDS)} seeds) ===\n")

    for cfg in CONFIGS:
        label = config_label(cfg)
        results.setdefault(label, [])
        for seed in SEEDS:
            if (label, seed) in done:
                print(f"{label:24s} seed={seed}  SKIPPED (already in {summary_path})")
                continue
            kwargs = dict(cfg)
            if kwargs.get("multi_critic"):
                kwargs["critic_model"] = critic_model
            log = inspect_eval(
                elicit(suite=SUITE, **kwargs),
                model=model,
                limit=LIMIT,
                temperature=TEMPERATURE,
                seed=seed,
                log_dir="./logs",
                max_connections=MAX_CONNECTIONS,
            )[0]
            acc = get_accuracy(log)
            log_path = str(log.location) if hasattr(log, "location") else None
            results[label].append((seed, acc, log_path))
            print(f"{label:24s} seed={seed}  accuracy={acc:.3f}  log={log_path}")
            _save_summary(summary_path, model, critic_model, results)

    _save_summary(summary_path, model, critic_model, results)
    print(f"\nSaved -> {summary_path}")


def main():
    print(f"{len(CONFIGS)} configs x {len(SEEDS)} seeds x {len(MODEL_PAIR)} models "
          f"= {len(CONFIGS) * len(SEEDS) * len(MODEL_PAIR)} total runs\n")
    for model, critic_model in MODEL_PAIR:
        run_for_model(model, critic_model)
    print("\nBoth models done. Next: Shapley attribution + interaction term "
          "computation over each model's results/shapley_sweep_gpqa_*.json "
          "(Phase 10 -- not yet built).")


if __name__ == "__main__":
    main()
