# Process Log

Running record of what was actually tried, what worked, what didn't, and
why decisions were made. Written close to when things happened. This is
the raw material for the polished writeup later — don't clean it up
retroactively, just keep appending.

---

## Phase 1 — Infrastructure setup

Set up the repo on top of Inspect AI (UK AISI's eval framework) rather
than building model clients, dataset loaders, and logging from scratch.
Decision rationale: the novel contribution is the measurement idea
(component attribution, budget substitution, cross-model transfer), not
plumbing that already exists and is used by serious evaluators (METR,
AISI). Building the plumbing from scratch would have signaled the
opposite of what a senior engineer/researcher does — reinventing solved
infrastructure instead of building the novel layer on top of it.

- `uv venv --python 3.12`, installed `inspect-ai` + `inspect_evals`
- GitHub repo created public, MIT license, `.gitignore` for `.env` and
  `.venv/`
- Directory skeleton scaffolded: `src/elicit/{models,components,budget,
  tasks,runner,attribution,analysis}`, `experiments/`, `results/`,
  `figures/`, `suites/`, `writeup/`
- Smoke test: `inspect eval inspect_evals/gsm8k --model openai/gpt-4o-mini
  --limit 20` → accuracy 0.85-0.90 range, confirmed the full stack
  (model interface, dataset loading, scoring, logging) works end to end

## Phase 2 — Vertical slice: prove the measurement loop works

Built `elicit_task.py` (task definition with toggleable solver
components) and `run_ablation.py` (sweep runner) as a minimal two-
component slice: `chain_of_thought()` and `self_critique()`, both
Inspect built-ins requiring no sandbox. Kept `shuffle=False` on every
dataset load from the start — non-negotiable for valid comparison, since
every config being compared has to see the identical samples in the
identical order.

**First real run (GSM8K, n=30):**
```
bare             accuracy=0.933
critique         accuracy=0.967   (delta vs bare: +0.033)
cot              accuracy=0.933   (delta vs bare: +0.000)
cot+critique     accuracy=0.967   (delta vs bare: +0.033)
```
Loop confirmed working end to end. Numbers not trustworthy yet at n=30 —
treated as a smoke test, not a result.

## Phase 3 — Finding a task suite with real headroom

**GSM8K at n=200:**
```
bare             accuracy=0.915
critique         accuracy=0.930   (+0.015)
cot              accuracy=0.940   (+0.025)
cot+critique     accuracy=0.930   (+0.015)   <- lower than cot alone
```
Ran McNemar's test (built `mcnemar_test.py`, paired significance test
using the fact that every config runs on identical samples). Results:
bare vs cot p=0.18, cot vs cot+critique p=0.75. Neither significant.
Root cause: GSM8K bare accuracy (91.5%) is too close to ceiling for this
model — too few discordant pairs (9-10 out of 200) for the test to have
power. This is a genuine methodological finding, not a failure: the task
was wrong for this model, not that scaffolding doesn't matter. Decision:
move to a harder suite. Kept the GSM8K result as documented evidence of
this discipline, didn't discard it.

**Attempted: MATH (EleutherAI/hendrycks_math, algebra subject).**
Bare accuracy at n=20 landed at 0.700 — good headroom. But manual
transcript review surfaced false negatives: mathematically correct
answers scored wrong due to superficial formatting differences —
`\frac{9}{7}` vs `\dfrac{9}{7}`, `[-2,7]` vs `x \in [-2,7]`, `5.5` vs
`\frac{11}{2}`. Naive string-normalization scorer wasn't catching
representation-equivalent answers.

Fix: tiered `_math_answers_match()` — (1) LaTeX macro cleanup
(`\dfrac`/`\tfrac` → `\frac`, strip `\left`/`\right`, strip a leading
"x \in " prefix), (2) sympy-based symbolic/numeric equivalence with a
manual `\frac{a}{b}` → `(a)/(b)` fallback (since the optional `antlr4`
LaTeX parser wasn't installed). Verified against all three reported
false negatives — all three now score correctly. Known residual
limitation, documented: this still won't catch every equivalence class
(e.g. genuinely different interval notations are correctly NOT
conflated; some multi-variable or set-builder expressions may still
fall back to string match). Real accuracy is likely somewhat higher than
raw scores suggest even after the fix — flagged as a validity caveat
for the methodology section rather than fully resolved.

**Decision: move to GPQA Diamond instead of continuing to harden the
MATH scorer.** Multiple-choice format sidesteps LaTeX equivalence
entirely — the "answer" is a single letter, so scoring is a straight
string match with no equivalence logic needed. Also: GPQA Diamond is
specifically designed to be hard even for frontier models (human expert
baseline ~69.7%), gated on HuggingFace specifically to reduce training
contamination (itself a relevant validity-hazard data point), and
already listed in the `inspect_evals` catalog.

**GPQA setup:**
- Required HF_TOKEN: accepted dataset terms on HF, generated a read-only
  token, added to `.env` (auto-loaded by Inspect the same way as
  `OPENAI_API_KEY`)
- Built a `gpqa` adapter: loads `Idavidrein/gpqa` (`gpqa_diamond`
  config), constructs 4-choice options (correct + 3 incorrect) with a
  **deterministic per-question shuffle** (seeded off the question text,
  not a global RNG) so every config/model/seed sees identical option
  lettering for a given question — required for equal-footing
  comparison
- `letter_match()` scorer: regex-extracts the letter after `ANSWER:`,
  straight string comparison against target

**Bug found and fixed: system prompt was leaking CoT into the "bare"
condition.** First GPQA smoke test (n=20) used a system prompt
containing "Reason through it" in every config, including `cot=False`.
Result: 0.700 accuracy, well above the published 35-45% GPT-4-class
range — the bare condition wasn't actually bare. Confirmed via output
token count (10,818 output tokens across 20 samples = real reasoning
happening even when `cot=False`).

Fix: split into `GPQA_SYSTEM_BASE` — "Answer directly... do not explain
your reasoning" — for the true bare condition, letting the existing
`chain_of_thought()` solver step be the only source of reasoning when
`cot=True`. Rerun at n=20 with the fix: accuracy 0.550, output tokens
dropped to 60 across 20 samples (~3 tokens/answer — confirms the model
was giving bare letter answers with no reasoning leaking through).
0.550 still above published range but within the wide CI at n=20
(stderr 0.114) — noted as not yet resolvable at this sample size,
deferred to the full run rather than over-interpreted.

## Phase 4 — First real ablation result (GPQA, cot + critique)

**n=200 (198, full Diamond set), temperature=0 (single seed):**
```
bare             accuracy=0.338
critique         accuracy=0.374   (+0.035)
cot              accuracy=0.409   (+0.071)
cot+critique     accuracy=0.384   (+0.045, lower than cot alone)
```
Bare accuracy (33.8%) finally landed in the published range — confirms
the system-prompt fix worked and this run measures something real.
McNemar (single run): bare vs cot p=0.098 — a real-looking effect
(+7.1pts) that doesn't clear α=0.05, limited by disagreement count (~60
discordant pairs) rather than the outcome being genuinely null. GPQA
Diamond is capped at 198 questions total, so raising `--limit` further
wasn't an option for more power.

**Decision: add temperature + multi-seed pooling to get more power
without more raw samples.** At temperature=0 the model is
near-deterministic, so repeated runs on the same 198 questions add no
new information. Set `TEMPERATURE=0.7` (real sampling variation, still
coherent) identically across every config being compared (never varying
temperature between compared configs — would confound the scaffold
effect with a temperature effect). Ran `SEEDS=[1,2,3,4,5]`. Updated
`run_ablation.py` to sweep seeds and save per-config log paths to
`results/ablation_summary.json`; extended `mcnemar_test.py` with a
`--pooled` mode that sums the 2x2 discordant-pair table across all 5
seeds before running one binomial test on the pooled total (documented
caveat: repeated seeds on the same fixed question set aren't fully
independent draws, so the resulting p-value is an approximation, not
textbook-exact). Also fixed a latent bug in `mcnemar_test.py`: the
scorer-name lookup was hardcoded to `"match"`, which would have silently
broken on the `boxed_match` and `letter_match` logs — generalized to try
a list of known scorer names.

**5-seed pooled run (n=198, temperature=0.7) — means:**
```
bare             mean=0.351  (seed stdev=0.016)
critique         mean=0.355  (seed stdev=0.016)   (+0.004)
cot              mean=0.386  (seed stdev=0.019)   (+0.035)
cot+critique     mean=0.383  (seed stdev=0.018)   (+0.032, lower than cot alone)
```
Tight seed-to-seed consistency (stdev 0.016-0.019) — the sampling is
behaving sensibly. Pattern replicates what was seen on GSM8K and the
single-seed GPQA run: critique alone does ~nothing; CoT is the real
driver; adding critique on top of CoT never adds to the gain and
consistently sits slightly below CoT alone. Third independent
observation of this interaction, across two different task types now.

**Status at time of writing: pooled McNemar significance test on bare
vs cot, and cot vs cot+critique, has been run/is being run against
`results/ablation_summary.json` — final p-values not yet confirmed
and written down here. This is the one open item before Phase 4 can be
marked complete.** Do not proceed to Phase 5 (tool use) until both
p-values are recorded here with their interpretation.

---

## Open item / next entry to add

- [ ] Record pooled McNemar p-value: bare vs cot
- [ ] Record pooled McNemar p-value: cot vs cot+critique
- [ ] Write one paragraph interpreting both in plain language
      (significant / trend / not significant), and note whether the
      three-run replication of the "critique doesn't stack with cot"
      pattern is now considered a supported finding or still just
      suggestive
- [ ] Then: begin Phase 5 (tool use, Docker sandbox)
