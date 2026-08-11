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

## Phase 2 follow-up — MATH scorer robustness: quantifying the residual gap

The MATH scorer fix documented under Phase 3 (sympy tier, three false
negatives caught and fixed) was never checked against a systematic sample
— just the three examples that happened to surface during setup. Went
back and did that properly, reusing the 5 existing MATH pilot logs
(`logs/2026-08-05T01-*`, `logs/2026-08-05T03-*`, n=20 each, bare config)
rather than spending new API budget on fresh runs.

**Method:** pulled every sample scored `boxed_match=I` across all 5 logs,
deduped by (model_answer, target) pair → 39 unique "incorrect" cases.
Manually read each one against the model's raw completion.

**Result: 12 of 39 (31%) were scorer bugs, not model errors.** Root
causes, all fixed in `elicit_task.py`:
- `\$40` vs `40` — the old `$`-strip did `s.replace("$", "")` on `\$40`,
  which only deletes the `$` character and leaves a dangling backslash
  (`\40`), so it silently never matched. Now strips `\$` explicitly first.
- Brace-less `\frac`/`\sqrt` shorthand (`\frac56`, `\frac9{5}`, `\sqrt7`,
  `\frac 12`) — the dataset mixes this with the fully-braced form
  interchangeably; the sympy converter only recognized `\frac{a}{b}`.
  Added `_brace_frac_sqrt_args()` to canonicalize both into braced form
  before comparison (5 cases fixed directly, plus this made an
  interval-union answer match via plain string equality once both sides
  normalized identically).
- Braced exponents + implicit multiplication (`x^{9}` vs `x^9`,
  `7(x-3)(x+3)` vs `7(x+3)(x-3)`) — plain `sympify()` can't parse either
  (Python reads `{9}` as a set literal, and implicit adjacent-parens
  multiplication is a syntax error). Added `x^{9}` → `x^(9)` rewriting
  plus a last-resort parse tier using sympy's own `parse_expr` with
  `implicit_multiplication_application` + `convert_xor` — sympy's
  purpose-built tolerant parser, not a hand-rolled regex, specifically to
  avoid mis-parsing ambiguous input into a wrong-but-valid expression.
- Coordinate-pair answers (`(1, 4.5)` vs `(1, \frac{9}{2})`) — `sympify()`
  returns a plain Python tuple for these, which doesn't support `-`, so
  the old code fell through to string comparison and missed it. Added
  element-wise symbolic comparison when both sides parse to tuples.

**Verification, both directions:**
- Re-ran all 39 previously-incorrect cases against the fixed matcher: 12
  now flip to correct (matches the false negatives found above), the
  remaining 27 stay incorrect. Read all 27 by hand — genuinely wrong
  model answers (numeric mismatches, different polynomials, wrong
  variable/expression, word-vs-number mismatches), not new bugs.
- Re-ran all 105 previously-*correct* cases from the same 5 logs against
  the fixed matcher as a regression check: **zero flipped to incorrect.**

**Headline number:** on the existing MATH pilot data, true bare accuracy
was understated by 12/39 ≈ 31% of what the scorer had been calling wrong
— i.e. a meaningful chunk of "model errors" in the pilot runs were
actually scorer errors.

**Caveat — this does not yet close the Phase 2 checkbox.** This was an
exhaustive review of every incorrect verdict already present in 5 small
pilot logs, not the ≥30-random-transcript spot-check with a false
*positive* rate that Phase 2 requires. It's a strictly narrower check: it
only looked at MATH, and only at samples already scored wrong (so it
can't catch the matcher being too *lenient* — a false positive where two
genuinely different answers get waved through as equivalent). GPQA's
`letter_match` still has no spot-check at all. Still need: a real random
30-transcript sample for both suites, checked for both directions of
error, before Phase 2 is actually done.

---

## Phase 4 — pooled McNemar results (GPQA)

Ran the two pending pooled tests against `results/ablation_summary.json`
(GPQA Diamond, n=198, 5 seeds, temperature=0.7):

```
python mcnemar_test.py --pooled results/ablation_summary.json bare cot
python mcnemar_test.py --pooled results/ablation_summary.json cot cot+critique
```

**bare vs cot:** mean accuracy 0.351 vs 0.386. Pooled 2x2 (275 total
disagreements across 5 seeds): bare-right/cot-wrong=120, cot-right/bare-
wrong=155. Exact McNemar p=0.0401 — **significant at p<0.05**. CoT is
reliably better than bare on GPQA, not just sampling noise.

**cot vs cot+critique:** mean accuracy 0.386 vs 0.383. Pooled 2x2 (197
disagreements): cot-right/critique-wrong=100, critique-right/cot-wrong=97
— almost exactly balanced. Exact McNemar p=0.887 — **not significant**.
The interaction finding (critique adds nothing on top of cot, and if
anything sits marginally below) now has a p-value behind it on GPQA, not
just an eyeballed delta: the two configs are statistically
indistinguishable. This is the third independent observation of the same
pattern (GSM8K pilot, single-seed GPQA, now pooled 5-seed GPQA), and this
time it's backed by a significance test rather than an eyeballed delta —
considered a supported finding on GPQA specifically, not yet cross-suite
since the MATH pooled sweep hasn't been run (see below).

As noted in mcnemar_test.py's own pooled-mode caveat: seeds share the
same fixed 198-question set, so treat both p-values as approximate rather
than textbook-exact independent draws.

**This closes the GPQA half of Phase 4's blocking checklist item.** The
MATH half is still open — `results/ablation_summary.json` only has a
`"suite": "gpqa"` run; the 5-seed × 4-config sweep has never been run on
MATH, only single-seed n=20 pilots (see Phase 2 follow-up above). Given
MATH_SUBJECT is currently `"algebra"` and is showing signs of being
near-ceiling for gpt-4o-class models (little room for scaffold components
to move the needle), don't spend the API budget on a full pooled MATH
sweep until the suite's headroom problem is resolved — see the open item
below.

## Open item / next entry to add

- [x] Record pooled McNemar p-value: bare vs cot — p=0.0401, significant
- [x] Record pooled McNemar p-value: cot vs cot+critique — p=0.887, not
      significant
- [x] Write one paragraph interpreting both in plain language — done
      above
- [ ] Fix MATH headroom (swap `MATH_SUBJECT` to a harder subject, and/or
      filter by the already-captured `level` metadata) before running
      the 5-seed pooled MATH sweep
- [ ] Run the ≥30-random-transcript spot check (`spot_check.py`, just
      added, not yet run against any log) for both GPQA and MATH,
      counting false positives AND false negatives — the other open
      Phase 2 requirement, separate from the MATH scorer's own
      exhaustive-incorrect-case review above
- [ ] Then: begin Phase 5 (tool use, Docker sandbox)
