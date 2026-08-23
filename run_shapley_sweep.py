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
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

from elicit_task import elicit

SUITE = "math"
LIMIT = 500  # intermediate_algebra has 903 available -- not capped like GPQA
# Targeted budget cut for the 8 of 16 configs with multi_critic=True --
# these are both the slowest runs in the sweep (multi_critic calls a
# SECOND model) and the one place power_analysis.py already showed the
# budget was disproportionate: Qwen's multi_critic main effect needs
# N=7416 (unreachable at ANY sane budget, full or cut) while gemma's
# needs only N=587. n=250 x 5 seeds = 1250 clears gemma's requirement
# with 2x margin, costs nothing on Qwen's (already unreachable at 2500
# too), and is still a real, non-trivial sample size for every
# interaction term that touches multi_critic. Applied to BOTH models
# uniformly (not just Qwen, which is the one that actually needed it) --
# never let generation config, including sample size, differ between
# directly-compared configs. See process_log.md's "cutting the sweep
# further" entry for the full reasoning and the required-N numbers this
# was checked against.
MULTI_CRITIC_LIMIT = 250
TEMPERATURE = 0.7
SEEDS = [1, 2, 3, 4, 5]
# 20, not the original 5 -- that value was calibrated against an OpenAI
# rate-limit incident (see process_log.md's "gpt-4o-mini MATH ablation"
# entry) that has nothing to do with DeepInfra's actual limits.
# Re-tested live rather than carried forward by habit: a 40-sample MATH
# run took 142.8s at max_connections=5, 94.0s at 20 (~1.5x faster), 103.1s
# at 40 (no further gain -- likely DeepInfra's own throughput ceiling,
# not our client-side limit). See process_log.md's runtime-investigation
# entry for the full story (this alone doesn't fix a 12-day sweep --
# running both models as separate parallel processes is the other half).
MAX_CONNECTIONS = 20
# Qwen3-32B (DeepInfra) defaults to max_tokens=65536, which exceeds its own
# 40960-token context window and 400s on every call unless capped explicitly
# -- found via a live smoke test before trusting anything (see
# process_log.md's "switch to DeepInfra" entry). 4096 comfortably covers
# every real per-turn output size measured so far on either suite (MATH's
# heaviest single-turn cot+critique data point was ~1725 tokens) and is
# applied uniformly to every model, not just the one that needed it --
# never let generation config differ between compared configs/models.
MAX_TOKENS = 4096
# Qwen3-32B has an internal "thinking" mode that fires on hard questions
# regardless of the system prompt's "do not explain your reasoning"
# instruction -- found via transcript inspection, not just the aggregate
# accuracy number, which looked fine on its own: a 20-sample GPQA pilot
# showed 2 of 8 inspected samples burning the full max_tokens budget on
# hidden reasoning with ZERO visible answer (real no-answer failures an
# earlier automated check incorrectly reported as 0/20), one sample
# leaking reasoning into the visible text AND breaking the required
# "ANSWER: X" format, and reasoning blocks up to 16,319 chars on samples
# that did produce a clean answer -- the same class of problem that got
# Qwen3.5-9B rejected earlier in this project (see process_log.md's
# "Qwen3.5-9B turned out to be unusable" entry), just less visible in the
# top-line accuracy. `chat_template_kwargs: {"enable_thinking": False}`
# (the standard vLLM/Qwen3 convention, confirmed working via a direct
# smoke test even though DeepInfra's own docs don't mention it) fixes
# this: output tokens on the same hard question dropped from 1977 to 5
# with the same correct answer. Confirmed harmless on gemma-3-27b-it too
# (extra fields are just ignored), so applied to every model uniformly.
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
# DeepInfra's "Flex" service tier (0.8x standard pricing, in exchange for
# "slower responses and occasional unavailability" -- explicitly meant
# for non-production/async work, which this sweep is) -- confirmed live
# per-model before applying anywhere, not assumed uniform: gemma-3-27b-it
# and Qwen2.5-72B-Instruct both returned in ~1s on Flex; Qwen3-32B hit
# the "occasional unavailability" case hard -- a single request took
# 1804s (30 min) before timing out. Applied selectively, not globally,
# for exactly this reason. Applies both when a model is the PRIMARY
# model being evaluated and when it's serving as the CRITIC for another
# model's multi_critic runs (gemma is critic for both other sweeps, so
# this matters for its critic role too, not just its own sweep).
#
# gemma removed 2026-08-23: with both other sweeps finished, gemma had
# zero remaining contention but was still stuck badly -- direct evidence
# from the sample-buffer db, not just slow logs: tool_use+multi_critic
# seed=4 finished all 250 samples in 10 minutes, seed=5 (same config)
# did only 10/250 samples in 10h47m before being killed. Same
# "occasional unavailability" failure mode already documented for
# Qwen3-32B on flex (one request took 30 min). Costs ~25% more on
# gemma's remaining calls but trades away a demonstrated 270x slowdown.
FLEX_TIER_MODELS = {
    "openai-api/deepinfra/Qwen/Qwen2.5-72B-Instruct",
}


def extra_body_for(model_name: str) -> dict:
    body = dict(EXTRA_BODY)
    if model_name in FLEX_TIER_MODELS:
        body["service_tier"] = "flex"
    return body

# (model string, critic model string for THIS model's multi_critic runs)
# DeepInfra, not Together -- switched for cost (see process_log.md's
# "switch to DeepInfra" entry). Neither original model (Qwen2.5-7B-
# Instruct-Turbo, gemma-3n-E4B-it) exists on DeepInfra's catalog, so this
# is a genuine model-selection redo, not a drop-in provider swap -- picked
# deliberately larger models (32B/27B, still cheap on DeepInfra) to also
# test the standing "maybe small models can't use these components"
# hypothesis from GPQA's null result.
# Tried microsoft/phi-4 (14B) as a third leg for a 3-tier scale
# comparison -- rejected: DeepInfra returns a hard 405
# "Tool calling is not supported for model: microsoft/phi-4" the moment
# tool_use=True is invoked, which blocks 8 of the 16 configs outright.
# Not a fixable bug, a real capability gap. See process_log.md's
# "phi-4 rejected -- no tool-calling support" entry.
#
# Also tried meta-llama/Llama-3.3-70B-Instruct-Turbo -- clean behavior
# and passes tool-calling, but a controlled n=20 timing test (same
# conditions as everything else here) showed it at 102.5s vs Qwen3-32B's
# 44.9s -- i.e. SLOWER than the current bottleneck, not comparable to it
# (an earlier, uncontrolled comparison had incorrectly suggested they
# were similar speed -- corrected after the user pushed back and asked
# for a real apples-to-apples test). Adding it as a third leg would have
# made the whole sweep slower, not just added a data point. Rejected for
# that reason, not a capability problem.
#
# Qwen/Qwen2.5-72B-Instruct passed every check: tool-calling confirmed
# working, controlled n=20 timing at 12.2s (faster than BOTH models
# already running, so it won't become the bottleneck), clean transcripts
# (proper \boxed{} only, no hidden reasoning -- same non-reasoning
# generation as our very first validated model, Qwen2.5-7B), and real
# headroom (0.300-0.350 across two separate n=20 checks). 72B is a
# genuine third scale tier against the 27B/32B pair already running.
# Critic model is gemma-3-27b-it, same reasoning as phi-4's attempt: fast,
# cheap, already validated, doesn't require touching the two sweeps
# already in progress.
MODEL_PAIR = [
    ("openai-api/deepinfra/Qwen/Qwen3-32B", "openai-api/deepinfra/google/gemma-3-27b-it"),
    ("openai-api/deepinfra/google/gemma-3-27b-it", "openai-api/deepinfra/Qwen/Qwen3-32B"),
    ("openai-api/deepinfra/Qwen/Qwen2.5-72B-Instruct", "openai-api/deepinfra/google/gemma-3-27b-it"),
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
                # multi_critic configs run at a smaller n -- see
                # MULTI_CRITIC_LIMIT's comment above for why. Each run's
                # actual n is recorded per-entry below (not just implied by
                # this default), since it now varies by config.
                "multi_critic_limit": MULTI_CRITIC_LIMIT,
                "temperature": TEMPERATURE,
                "seeds": SEEDS,
                "components": TOGGLES,
                "results": {
                    label: [
                        {"seed": s, "accuracy": a, "log": p, "limit": n}
                        for s, a, p, n in runs
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
        nan_count = 0
        for label, runs in prior.get("results", {}).items():
            # r.get("limit", LIMIT): entries saved before this per-config
            # limit existed don't have the field -- they were genuinely all
            # run at the (then-uniform) LIMIT, so that's the correct
            # fallback, not a guess.
            #
            # NaN entries are dropped here, not carried over -- a NaN
            # means that (config, seed) FAILED (e.g. the DeepInfra
            # account-balance outage that hit all three sweeps at once),
            # and the original "any recorded entry counts as done"
            # resume logic would otherwise treat a failure as permanently
            # finished, silently leaving corrupted data in place forever.
            # Dropping it from both `results` and `done` means a normal
            # relaunch naturally retries exactly the failed runs and
            # appends a fresh entry -- no manual per-entry patching
            # needed, and no risk of ending up with two entries for the
            # same (config, seed) once the retry succeeds.
            results[label] = []
            for r in runs:
                if r["accuracy"] != r["accuracy"]:  # nan check
                    nan_count += 1
                    continue
                results[label].append(
                    (r["seed"], r["accuracy"], r["log"], r.get("limit", LIMIT))
                )
                done.add((label, r["seed"]))
        if done:
            print(f"Resuming {summary_path}: {len(done)} (config, seed) runs "
                  f"already complete, skipping those.")
        if nan_count:
            print(f"Dropped {nan_count} NaN (failed) entries -- these will be "
                  f"retried this run.\n")

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
            run_limit = LIMIT
            if kwargs.get("multi_critic"):
                kwargs["critic_model"] = critic_model
                # self_critique(model=critic_model) resolves the critic via
                # its own unconfigured get_model() call, so it never
                # inherits inspect_eval()'s max_tokens/extra_body below --
                # has to be passed separately or the critic model hits the
                # same overflow bug fixed for the primary model (see
                # elicit_task.py's multi_critic() docstring).
                kwargs["critic_config"] = GenerateConfig(
                    max_tokens=MAX_TOKENS, extra_body=extra_body_for(critic_model),
                )
                run_limit = MULTI_CRITIC_LIMIT
            log = inspect_eval(
                elicit(suite=SUITE, **kwargs),
                model=model,
                limit=run_limit,
                temperature=TEMPERATURE,
                seed=seed,
                log_dir="./logs",
                max_connections=MAX_CONNECTIONS,
                max_tokens=MAX_TOKENS,
                extra_body=extra_body_for(model),
                # Default fail_on_error=True aborts the ENTIRE run (all
                # 500/250 samples -> accuracy=NaN) the moment a single
                # sample errors -- confirmed this is exactly why
                # tool_use's context-window-overflow failures (see
                # elicit_task.py's tool_use comment) were wiping out
                # whole seeds instead of just costing one wrong answer.
                # score_on_error scores the errored sample as incorrect
                # instead of excluding it, so n stays the intended
                # sample size. Pure harness robustness -- doesn't change
                # what tool_use measures, unlike a tool-call round cap
                # would (see process_log.md's "declined" entry on that).
                fail_on_error=0.02,
                score_on_error=True,
            )[0]
            acc = get_accuracy(log)
            log_path = str(log.location) if hasattr(log, "location") else None
            results[label].append((seed, acc, log_path, run_limit))
            print(f"{label:24s} seed={seed}  accuracy={acc:.3f}  n={run_limit}  log={log_path}")
            _save_summary(summary_path, model, critic_model, results)

    _save_summary(summary_path, model, critic_model, results)
    print(f"\nSaved -> {summary_path}")


def main():
    # Optional sys.argv[1] = 0 or 1 selects a single model from MODEL_PAIR
    # to run in THIS process, so two separate OS processes (launched
    # separately, e.g. `python run_shapley_sweep.py 0` and `... 1`) can run
    # concurrently -- each with its own independent Inspect runtime state.
    # Confirmed live that running two models this way works cleanly (no
    # rate-limit conflicts): Inspect's `eval_async` explicitly forbids
    # concurrent calls WITHIN one process ("Multiple concurrent calls to
    # eval_async are not allowed"), which is why this is two processes,
    # not asyncio concurrency inside a single run of this script. No arg
    # (the default) runs both models sequentially in this one process, as
    # before.
    if len(sys.argv) > 1:
        pairs = [MODEL_PAIR[int(sys.argv[1])]]
    else:
        pairs = MODEL_PAIR

    print(f"{len(CONFIGS)} configs x {len(SEEDS)} seeds x {len(pairs)} model(s) "
          f"= {len(CONFIGS) * len(SEEDS) * len(pairs)} total runs\n")
    for model, critic_model in pairs:
        run_for_model(model, critic_model)
    print("\nDone. Next: Shapley attribution + interaction term "
          "computation over each model's results/shapley_sweep_"
          f"{SUITE}_*.json (shapley_attribution.py).")


if __name__ == "__main__":
    main()
