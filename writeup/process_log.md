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

## Phase 3 revisit — MATH headroom: algebra's ceiling, subject comparison, switch to intermediate_algebra

The open item from the previous entry ("fix MATH headroom") turned into a
much bigger investigation than expected — worth its own entry.

**Algebra's headroom was never actually validated past n=20.** Ran a
fresh bare-config batch at n=1000 (the algebra test split has 1187 total,
so this isn't near the dataset's own cap the way GPQA is). Result:
**accuracy 0.909** — right next to GSM8K's rejected 0.915. The n=20 pilot
number (0.700) that justified adopting MATH in the first place was badly
misleading; small-n pilots are not a reliable basis for suite selection,
full stop.

**Used the fresh n=1000 data to find more scorer bugs, this time on data
the earlier fix had never seen.** Pulled all 91 unique incorrect verdicts,
confirmed the earlier fix introduced zero regressions (0 of the 91 flip
under the already-fixed matcher — consistent, not a fix-application bug),
then manually read all 91 by hand. Found **15 more confirmed false
negatives**, all fixed in `elicit_task.py`:
- `\text{...}` unit/label suffix never stripped (7 cases — `118` vs
  `118\text{ dollars}`, `85` vs `85\text{ feet}`, etc). Added
  `_strip_text_wrapper()`: unwraps `\text{...}` when it's the *entire*
  answer (keeps contents, e.g. `\text{Evelyn}` → `Evelyn`), strips it as a
  discardable label when it's a suffix on a numeric answer (also strips a
  trailing unit-exponent like the `^2` in `440\text{ cm}^2`, or a bare
  `^2` is left behind corrupting the number).
- **Pre-existing bug, not caught by the original fix**: the `\dfrac`/
  `\tfrac` → `\frac` regex was `\\d?frac` — only ever made the "d"
  optional, so despite the docstring's own claim, `\tfrac` was silently
  never converted. Fixed to `\\[dt]?frac`.
- Comma thousands-separators in dollar figures (`115000` vs `\$115,000`,
  3 cases) — collapsed only when the whole cleaned answer fullmatches a
  grouped-number pattern, specifically to avoid corrupting a genuine
  multi-value list like `4,6,14,15`.
- `+\infty` vs `\infty` in interval notation (2 cases) — now normalized.
- `\sqrt` followed by a space before its digit (`\sqrt 2` vs `\sqrt{2}`,
  1 case) — the earlier brace-fix regex required immediate adjacency.
- Nested-brace `\frac` arg (`\frac{\sqrt6}3` vs `\frac{\sqrt{6}}{3}`,
  1 case) — the old regex-based brace-fixer silently failed to match
  (and therefore skipped fixing the brace-less *sibling* arg) whenever
  either arg had nesting in it. Rewrote `_brace_frac_sqrt_args()` from a
  single regex into an explicit balanced-brace scanner
  (`_consume_balanced` / `_consume_frac_arg`), which handles arbitrary
  nesting correctly instead of patching one more special case.

Verified both directions again: all 15 flip to correct, the other 76 of
91 incorrect cases read by hand as genuinely wrong (including one
ambiguous-looking polynomial factorization, `(-4x²+2x+1)(4x²-1)` vs
`(-4x²+x+1)(4x²+x+1)`, confirmed NOT equal via `sympy.expand`). Regression
check across all 522 previously-correct verdicts (417 fresh + 105 from
the earlier 5-log check): **zero regressions.**

**Subject comparison, because algebra's ceiling meant a different subject
was needed anyway.** Hendrycks MATH has 7 subjects; piloted all of them
bare-config, gpt-4o-mini, n=40 each (n=500 for the two best):

| subject | n=40 | n=500 | notes |
|---|---|---|---|
| algebra | — | 0.909 (n=1000) | ceiling, rejected |
| number_theory | 0.875 | — | ceiling-adjacent, rejected |
| counting_and_probability | 0.575 | — | clean at n=40, not pursued further |
| **intermediate_algebra** | 0.600 | **0.552** | clean both times |
| precalculus | 0.450 raw / ~0.525 adjusted | — | 3 new bugs: degree-symbol suffix, 2× matrix (`\begin{pmatrix}`) notation — matrices are a structural 20% of the subject (8/40), not a rare tail case, so real matrix-comparison support would be needed to trust it |
| **geometry** | 0.425 | **0.555** | looked like the best headroom at n=40 — mostly small-sample noise, converges to a statistical tie with intermediate_algebra at n=500 |

**geometry vs intermediate_algebra at n=500 — tied on the headline number,
but not on scorer risk.** Triaged both n=500 runs: flagged every incorrect
pair with string-similarity > 0.5 as a false-negative candidate (56 for
intermediate_algebra, 57 for geometry) and read all 113 by hand.
intermediate_algebra: **zero** false negatives — every flagged pair,
including tricky-looking ones (`[2,∞)` vs `(2,∞)`, different bracket =
genuinely different set; `326680` vs `327680`, a real near-miss, not a
formatting issue) turned out to be a genuinely different answer.
geometry: **5 false negatives**, three of them the same root cause —
**`\pi` has no symbolic handling anywhere in `_sympy_equivalent`**
(`\frac{9}{2}\pi` vs `\frac{9\pi}{2}`, and `72\sqrt3\pi` vs `72\pi\sqrt3`,
same value, different bracketing/ordering that only a real `\pi → pi`
sympy mapping would resolve) — plus one percent-sign case (`33` vs
`33\%`, the second time this exact pattern has shown up, after also
appearing once in the algebra n=1000 data) and one mixed-number case
(`\frac{25}{13}` vs `1\frac{12}{13}`, i.e. 1 + 12/13 = 25/13, not
currently parsed at all). Adjusted geometry accuracy ≈0.566 — still
statistically tied with intermediate_algebra's 0.552.

**Decision: `MATH_SUBJECT = "intermediate_algebra"`.** Same headroom as
geometry once both are properly measured, zero outstanding scorer gap
(vs. geometry's real, recurring π-handling gap), and a larger total pool
(903 vs geometry's hard-capped 479) if more samples are ever needed.
Percent-sign and mixed-number handling are left unfixed for now — both
are real but lower-frequency patterns (2 confirmed instances total across
two different subjects); revisit if they keep recurring once the full
suite is in regular use.

**False-positive check on intermediate_algebra (the other open Phase 2
requirement).** Rather than a blind random sample, took the highest-risk
subset instead: every "correct"-marked case in the n=500 run where the
raw model-answer and target strings actually differed (i.e. the
equivalence logic had to do real work to call them equal) — 54 of 276
correct verdicts. Read all 54 by hand. **Zero false positives** — every
match is genuinely correct (spacing normalization, brace-less `\frac`
fixes, `\tfrac`, `\text{}` unwrap, `+\infty`, and one legitimate
sympy-verified term-reordering, `-5-3\sqrt5` vs `-3\sqrt5-5`, all doing
exactly what they're supposed to and nothing more).

This targeted approach is narrower than a true random sample in one way
(it can't catch a false positive that happens to have an identical raw
string to its target, though that category is close to definitionally
safe) but is a strictly harder test of the equivalence *logic* than a
random sample would be, since it concentrates entirely on the cases where
that logic actually did something. Combined with the exhaustive
false-negative reviews above, this is the most scrutinized any suite's
scorer has been in this project so far — intermediate_algebra is in
better shape at adoption time than GPQA or algebra ever were.

---

## Phase 4 — MATH ablation: OpenAI credit exhaustion, and a Together model-selection detour

Kicked off the actual point of everything above — the full 4-config ×
5-seed ablation on `intermediate_algebra` (n=500), mirroring the GPQA
protocol exactly (`run_ablation.py`, `SUITE = "math"`, `MODEL =
"openai/gpt-4o-mini"`, unchanged).

**OpenAI account ran out of credits mid-run.** `bare` seed 1-4 completed
cleanly (accuracy 0.548, 0.524, 0.538, 0.572 — consistent with the 0.552
n=500 pilot), then seed 5 onward hit `RateLimitError 429
credit_balance_exhausted` and sat retrying with exponential backoff
(30 min waits) until the run was killed. Nothing lost — the 4 completed
seeds' `.eval` logs are safe in `logs/`, and `results/ablation_summary_
math.json` is only written at the very end of the script, so the partial
run never got a chance to overwrite anything.

**Fixed a real risk in `run_ablation.py` while switching suites, before
running anything:** it previously wrote unconditionally to
`results/ablation_summary.json`, which already held the GPQA run — a MATH
run would have silently clobbered it. Changed the output path to be
suite-specific (`results/ablation_summary_{SUITE}.json`), and updated the
script's own printed follow-up command to match. This is the exact
failure mode Phase 4's acceptance criteria warns about ("don't let a MATH
run silently overwrite the GPQA one").

**Decision: switch to Together rather than wait for OpenAI credits.**
Added `TOGETHER_API_KEY` to `.env`. Confirmed Inspect's built-in Together
provider needs exactly that env var name (checked the provider source
directly rather than assuming) — no code/wiring changes needed, just a
model id and `together/<id>` as the model string.

**Tried `Qwen/Qwen2.5-7B-Instruct-Turbo` first** (the id already hinted at
in `run_ablation.py`'s old comment). Confirmed live via a n=5 smoke test,
then an n=40 headroom pilot on intermediate_algebra: **0.525 accuracy** —
close to gpt-4o-mini's 0.552, no ceiling/floor problem for this model
either. Launched the full ablation on it.

**Interrupted mid-run to evaluate `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-
397B-A17B` instead**, per a direct request. Neither name matches any
Qwen generation in training knowledge (Qwen → 1.5 → 2 → 2.5 → 3), so
checked Together's live `/v1/models` API rather than assuming either a
typo or a hallucination — **both are real, currently-hosted models**,
released after the assistant's January 2026 knowledge cutoff. Pricing
pulled from the same API response:

| model | input $/1M | output $/1M |
|---|---|---|
| Qwen2.5-7B-Instruct-Turbo | 0.30 | 0.30 |
| Qwen3.5-9B | 0.17 | 0.25 |
| Qwen3.5-397B-A17B | 0.60 | **3.60** |

397B-A17B's output pricing is 12x the small model's — a full 20-run
ablation would cost roughly $3 on Qwen3.5-9B vs **~$40 on the 397B model**,
against the README's own stated $60 total budget for the eventual full
cross-model study. Also: 9B+397B is exactly the same-family small/large
pair Phase 8 (cross-model transfer) will eventually need — using both now
would mean jumping ahead of Phases 5-7 in the roadmap. **Decision: use
only Qwen3.5-9B for now**, save the pair for when Phase 8 actually comes
up.

**Qwen3.5-9B turned out to be unusable at a reasonable token budget.**
n=40 pilot at default max_tokens: **0.050 accuracy** — implausibly low
for a model newer than Qwen2.5-7B. Investigated rather than accepting the
number: 38/40 samples had `stop_reason: max_tokens` with a **completely
empty completion** (0 chars). Raised `max_tokens` to 16000 and re-tested
at n=10: accuracy recovered to 0.4, but 6/10 samples *still* had no
`\boxed{}` (one still hit the 16000 ceiling with only 3,145 chars of
visible output), and cost ballooned (128,666 tokens for just 10 samples,
~12.7k tokens/sample vs Qwen2.5-7B's ~850/sample). Checked the raw
message object for a separate hidden-reasoning field (the DeepSeek-R1/
`<think>`-tag pattern) — none present; this model is just extremely
verbose in its visible answer content, not hiding tokens elsewhere.

**Side finding, not yet acted on:** this surfaced that `MATH_SYSTEM` (the
system prompt used for every MATH config, `cot` toggle or not) has always
said *"Show your reasoning, then give the final answer..."* — MATH never
got the same bare-vs-cot system-prompt split that GPQA's
`GPQA_SYSTEM_BASE`/`GPQA_SYSTEM` did after the CoT-leak bug (see Phase 3
above). This didn't cause visible problems for gpt-4o-mini or Qwen2.5-7B
(concise enough to stay within budget regardless), but it means MATH's
"bare" condition has technically never been fully reasoning-free for any
model — worth fixing before treating MATH's bare-vs-cot delta as
precisely comparable to GPQA's. Flagged, not fixed — out of scope for
unblocking the current ablation.

**Decision: reverted to `Qwen/Qwen2.5-7B-Instruct-Turbo`.** Already
validated clean (smoke test + n=40 pilot, zero boxed-answer failures,
reasonable token usage) and unblocks the actual goal — the MATH ablation
run — without an open-ended debugging detour into a verbose reasoning
model. Re-launched the full 4-config × 5-seed × n=500 ablation on it.

**Standing caveat, restated:** the eventual GPQA-vs-MATH replication
check is now comparing gpt-4o-mini (GPQA) against Qwen2.5-7B-Instruct-
Turbo (MATH) — a suite change confounded with a model change. Re-run MATH
on gpt-4o-mini once OpenAI credits are back for a clean comparison;
treat this Together run as a preview until then.

## Phase 4 — MATH ablation results (Qwen2.5-7B, intermediate_algebra): the opposite pattern from GPQA

Full 4-config × 5-seed × n=500 run completed cleanly (all 20 runs, zero
errors), saved to `results/ablation_summary_math.json` (confirmed
separate from GPQA's `results/ablation_summary.json` — the overwrite-risk
fix above worked).

```
bare             mean=0.526  (seed stdev=0.014)
critique         mean=0.489  (seed stdev=0.028)   (-0.038)
cot              mean=0.492  (seed stdev=0.008)   (-0.034)
cot+critique     mean=0.453  (seed stdev=0.045)   (-0.074)
```

Ran the same two pooled McNemar tests as GPQA:

**bare vs cot:** pooled 2x2 (490 total disagreements across 5 seeds):
bare-right/cot-wrong=288, cot-right/bare-wrong=202. Exact McNemar
**p=0.0001 — significant**. But the direction is reversed from GPQA:
**CoT makes this worse**, not better.

**cot vs cot+critique:** pooled 2x2 (584 disagreements): cot-right/
critique-wrong=341, critique-right/cot-wrong=243. Exact McNemar
**p=0.0001 — significant**. Critique makes it worse again, on top of an
already-worse cot.

**This is a real, well-powered effect (490 and 584 discordant pairs off
~2,500 samples per config each — not a small-sample fluke), and it is the
opposite pattern from GPQA in every respect**: on GPQA, cot helped
(+7.1pts→+3.5pts pooled, p=0.04) and critique was statistically neutral
on top of it (p=0.887); here, cot hurts and critique hurts further, both
at p=0.0001. Read naively, this is exactly the kind of comparison-flip /
non-transferring-attribution result the whole project exists to find.

**But it cannot be attributed to suite vs. model yet — it's confounded.**
GPQA ran on gpt-4o-mini; this MATH run is on Qwen2.5-7B-Instruct-Turbo
(the OpenAI-credit-exhaustion detour above). Two candidate explanations,
not yet distinguished:
1. A genuine suite effect — MATH's `MATH_SYSTEM` prompt already says
   "show your reasoning" in the bare condition (the prompt-split gap
   flagged above), so `cot`'s marginal step may be adding redundant or
   conflicting reasoning instructions rather than eliciting reasoning
   that wasn't there before, unlike GPQA where bare was verified
   genuinely reasoning-free after the CoT-leak fix.
2. A genuine model effect — Qwen2.5-7B may just be worse at using
   extended reasoning/self-critique productively than gpt-4o-mini,
   independent of which suite it's tested on.
Cannot tell which without re-running MATH on gpt-4o-mini (isolates the
model variable) and/or fixing the `MATH_SYSTEM` prompt split (isolates
the suite-design variable) — both already open items below. Do not
report this as a confirmed cross-suite finding until at least one of
those is done.

## Phase 3/4 — MATH_SYSTEM prompt-leak fix: confirmed, and the effect size is large

Fixed before spending any more compute on MATH, per external review of
this process log (correctly pointed out that re-running on gpt-4o-mini
without fixing the prompt first would just re-confound a different
pairing) — killed the gpt-4o-mini ablation that was already running on
the leaky prompt rather than let it finish and waste the budget on data
that would need to be discarded anyway.

**Fix, mirroring GPQA's identical fix exactly:** `MATH_SYSTEM` used to say
"Show your reasoning, then give the final answer..." *unconditionally*,
for every config regardless of the `cot` toggle. Changed it to "Give
ONLY the final answer, wrapped in `\boxed{...}`... Do not show your work
or explain your reasoning" — the fixed prompt used for every config, with
`chain_of_thought()` (added only when `cot=True` in `build_solver()`) as
the sole source of reasoning, exactly matching how `GPQA_SYSTEM_BASE`
already works.

**Validated the same way GPQA's fix was validated — output token counts,
not just accuracy:** n=20 pilot on gpt-4o-mini, intermediate_algebra:

| condition | avg output tokens/sample | accuracy |
|---|---|---|
| bare | 63 | 0.250 |
| cot | 518 (8x more) | 0.400 |

`\boxed{}` extraction still works with the new prompt (0/20 failures
either condition) — the model doesn't need to be told to "show work" to
know to wrap its final answer.

**The effect size is large, not cosmetic.** The old (leaky) bare
condition measured 0.552 accuracy at n=500 on this exact suite+model. The
properly-bare condition lands at ~0.25 — the leaked reasoning instruction
was roughly *doubling* measured bare accuracy. This confirms the
confound was real and substantial, not a theoretical nitpick.

**Consequence, stated explicitly per the review's point 2: every MATH
result generated before this fix is provisional, not just the eventual
model comparison.** That includes:
- The full Qwen2.5-7B-Instruct-Turbo ablation
  (`results/ablation_summary_math_qwen2.5-7b-instruct-turbo.json`) — the
  "cot hurts, critique hurts more" finding was measured with a
  contaminated bare condition and needs to be treated as unconfirmed
  until re-run.
- The 7-subject headroom comparison (algebra's 0.909 ceiling, geometry
  vs intermediate_algebra at n=500, etc.) — subject *rankings* are
  probably still roughly informative since every subject was measured
  with the same leak applied uniformly, but none of the absolute
  accuracy numbers can be trusted anymore.

Not re-running the full subject comparison right now (would be a large
compute spend to re-verify a ranking that's plausibly still correct) but
flagging it here so nobody cites those absolute numbers later without
this caveat. Proceeding straight to the gpt-4o-mini ablation with the
fixed prompt, since that's the one both this and the GPQA comparison
actually depend on.

---

## Phase 4 — gpt-4o-mini MATH ablation: three interruptions and a checkpointing fix

Getting the clean gpt-4o-mini re-run (fixed `MATH_SYSTEM` prompt, needed
for a real GPQA comparison) actually finished took four launches. Logging
the failure modes and fixes since they're real infrastructure lessons,
not just noise.

**Interruption 1 — a genuine OpenAI rate limit, distinct from the earlier
credit exhaustion.** First relaunch after the prompt fix: `bare` completed
cleanly, all 5 seeds (0.224, 0.240, 0.252, 0.234, 0.242 — consistent with
each other and with the n=20 validation pilot's 0.250). Then `critique`
seed 1 hit `RateLimitError 429 rate_limit_exceeded` (not
`credit_balance_exhausted` — a different account limit, throughput not
budget) partway through its 500 samples, and got stuck retrying with
exponential backoff (up to 1800s between attempts, 12+ retries) for over
an hour without recovering before being killed.

**Fix: capped concurrency.** Inspect's default `max_connections` is unset
(effectively ~10 concurrent requests), which was likely sustaining enough
request pressure that individual retries never found a clear window.
Added an explicit `MAX_CONNECTIONS = 5` constant to `run_ablation.py`,
threaded through to every `inspect_eval()` call. Validated with a smoke
test (n=10, `critique` config — the one that had failed) before trusting
it: completed cleanly, no rate-limit errors, though noticeably slower
(~3min for 10 samples vs. bare's much faster pace, since critique is a
2-turn generation).

**Interruptions 2 and 3 — the laptop going to sleep, not a code or API
issue.** Relaunched with the concurrency cap: first attempt produced a
completely empty output file and zero completed samples within minutes
(`status: started`, 0 results) — died almost immediately, no error
recorded anywhere. Relaunched again: this time `bare` seeds 1-4 completed
cleanly (0.240, 0.244, 0.240, 0.244, zero errors) before stopping
silently with no error output, right as it would have moved into
`critique`. Confirmed with the user afterward: the laptop had gone to
sleep both times, killing the background process outright — not
something fixable from the code side, but worth knowing the failure
signature (empty/truncated output, zero error lines, process just gone)
so it's not mistaken for a rate limit or API issue next time.

**Fix: checkpoint/resume support added to `run_ablation.py`.** Three
interruptions in a row each meant re-running `bare` from scratch — real
wasted API spend on a component that had already succeeded twice. Added
`_save_summary()` and called it after **every individual seed**, not just
at the end; and on startup, if a summary file for this exact suite+model
already exists, load it, build a set of already-completed `(config,
seed)` pairs, and skip those instead of re-running them. This makes any
future interruption resume from wherever it left off rather than
restarting the whole 20-run sweep.

**Current status: relaunched a fourth time with checkpointing in place,
in progress.** First checkpoint already written
(`results/ablation_summary_math_openai-gpt-4o-mini.json`, `bare` seed 1 =
0.244, consistent with prior runs). Will update this entry with full
results and the pooled McNemar tests once it completes.

## Phase 5 — tool use + HumanEval: code implementation done, empirical steps still pending

Built out the Phase 5 code while the gpt-4o-mini ablation ran in the
background, specifically because none of this needs the OpenAI API and
so doesn't compete with it for rate-limit budget.

**Docker: confirmed working.** Binary was already installed but the
daemon wasn't running (`docker run hello-world` initially failed with
"Cannot connect to the Docker daemon"). Started Docker Desktop; reran
`hello-world` successfully once it was up.

**`tool_use` toggle added to `build_solver()`.** Wired via Inspect's
built-in `use_tools(python())`, inserted before `generate()` — Inspect's
`generate()` handles the tool-call/tool-result loop internally once a
tool is registered, no custom looping logic needed. Threaded through
`elicit()`'s task signature as a new `tool_use: bool` parameter.

**`humaneval` adapter wired up for real** (was registered but commented
out, with `code_execution_match` raising `NotImplementedError`):
- Dataset ID needed fixing: `"openai_humaneval"` no longer resolves on
  current HuggingFace (unnamespaced repo IDs were deprecated) — confirmed
  via `HfApi().list_datasets(search="humaneval")` and found the correct
  current path, `openai/openai_humaneval`.
- Implemented `code_execution_match` for real: writes the model's
  completed function plus HumanEval's own `check(candidate)` test
  harness to a file inside the sandbox, executes it via
  `sandbox().exec()`, scores on the actual pass/fail result — not a text
  match, which is the entire reason this suite exists (it's the one
  place `tool_use` and later `planning` have something real to act on,
  per Phase 5's own rationale in phases.md). Added a markdown code-fence
  stripper (`_strip_code_fence`) since chat models routinely wrap answers
  in ` ```python ` blocks even when explicitly told not to — a real
  formatting habit, not a hypothetical edge case. 30s execution timeout
  guard in case the model's code has an infinite loop.
- `TaskAdapter` gained an optional `sandbox` field, threaded through to
  the `Task` constructor; `humaneval` is the only adapter that sets it.

**Verified everything except live model behavior, all without touching
the OpenAI API:** `elicit(suite="humaneval", tool_use=True)` builds a
`Task` with `sandbox='docker'` set correctly; the dataset loads (164
samples — matches HumanEval's known size); sample structure is right
(input = function signature/docstring, target = the `check(candidate)`
test string, metadata carries `entry_point`); solver step counts are
correct for every toggle combination (4 steps with `tool_use=True`: system
message, cot, use_tools, generate; 2 steps bare).

**Explicitly NOT done yet — the phase's own empirical requirements**,
deferred until the API isn't tied up with the MATH ablation:
- n=20 smoke test confirming the sandbox actually spins up, the model
  actually invokes the `python()` tool at least once (checked in
  transcripts, not assumed), and the scorer reads real execution results
  correctly
- Full ablation on `humaneval`: 4-config baseline (bare/cot/critique/
  cot+critique) for a same-footing comparison, then `tool_use` added as a
  5th toggle dimension (2⁴=16 configs, 5 seeds)
- Pooled McNemar: bare vs tool_use, cot+tool_use vs cot
- Transcript spot-check for actual tool-invocation rate on questions
  where it should plausibly help — not just "the code exists," a real
  rate needs to be reported per Phase 5's acceptance criteria

---

## Phase 6 — retrieval corpus v2: two real bugs in the Wikipedia category-fetch pipeline

Rebuilding `suites/retrieval_corpus.jsonl` via Wikipedia category
membership (per the decision in the prior entry) instead of a random
sample. Design: subject-specific categories (e.g. `Category:Quantum
mechanics`, not the broad `Category:Physics`, which mostly contains
biographical/institutional subcategories rather than concepts), one
level of subcategory recursion with an exclude-pattern filter
(`people`, `by country`, `list of`, etc.), capped at `MAX_TITLES=4000`
candidate article titles, then article text fetched directly from
Wikipedia's own API (`action=query&prop=extracts`) rather than by
streaming/filtering the full 6.4M-article HF `wikimedia/wikipedia` dump
by title (would require scanning a large fraction of the dump to find a
comparatively small target set).

**Bug 1 — Wikipedia's own rate limit.** First run hit `429 Too Many
Requests` on the very first category, partway through subcategory
traversal. Root cause: pacing (a `time.sleep`) only existed in the
extract-fetching loop, not in the category-traversal calls, which fan
out into far more requests per category than extraction does (each
category can have dozens of subcategories, each needing its own API
call). Fix: centralized ALL Wikipedia API calls through one `_api_get()`
function with built-in 429 handling (respects `Retry-After` if present,
exponential fallback otherwise, up to 6 attempts) and steady baseline
pacing — covers category traversal and extraction alike, rather than
requiring pacing to be re-added at every call site individually.

**Bug 2 — a MediaWiki API limit that silently discarded 95% of
articles.** After fixing the rate limit, the run completed but produced
only 195 docs from 4000 candidate titles — a loss rate far too high to
be genuine redirects/missing pages. Investigated rather than accepting
it: the API returns a warning (easy to miss without printing it
explicitly, which the original code didn't) —
`"exlimit" was too large for a whole article extracts request, lowered
to 1"` — meaning a FULL-article `explaintext` extract silently caps
`exlimit` to 1 regardless of what's requested, so each 20-title batch
was really only fetching 1 real article and treating the other 19 as if
they'd failed. Confirmed by re-querying one of the "failed" titles
directly and getting real content back. Fix: added `exintro=1` (intro
section only, not the full article body) to the extract request, which
batches cleanly with no limit warning — verified on a 4-title test batch,
all 4 returned real content (471–2322 chars each, comfortably above the
1500-char `PASSAGE_CHARS` truncation already in place, so nothing is
actually lost by not fetching full articles). This also explains most of
Bug 1's excess request volume: with only 1 real doc per batch, far more
round-trips were needed for the same yield than intended.

**Status: rebuild relaunched with both fixes, in progress** (interrupted
once more mid-run by the same laptop-sleep issue documented in the Phase
4 entry — relaunched again, no checkpointing on this script yet since
each run is short enough that a full restart is cheap, unlike the
multi-hour ablation). Will log final corpus size and the rerun
contamination-check verdict once it completes.

## Phase 7 — planning component: code implemented

Built the `planning` toggle's code while the MATH ablation and corpus
rebuild both continued in the background — pure code, zero API calls,
same reasoning as the earlier Phase 5/6 setup work.

**Design: a genuinely separate generation turn, not a relabeled
`chain_of_thought()`.** phases.md's own Phase 7 criteria specifically
warns against that shortcut. Implementation: `planning()` appends a
`ChatMessageUser` asking for a short numbered plan (`PLANNING_PROMPT`,
"output ONLY the numbered plan, do not solve yet"), then calls Inspect's
`generate()` directly inside the solver — which appends the resulting
plan as a real assistant turn to `state.messages` — before the main
`generate()` step (already in `build_solver()`'s pipeline) runs.

Chose this over the approach `retrieval()` uses (splicing text into
`state.user_prompt.text`) specifically because Phase 7's smoke test needs
to verify two things from transcripts: (a) plans differ per question, and
(b) the final answer actually references the plan. With the plan as a
real prior conversation turn, both are directly readable from any
transcript viewer (Inspect's own `inspect view`, or `read_eval_log`) —
the plan is literally an earlier turn the model can see when generating
its answer, not something to infer indirectly from prompt text that was
never actually part of the exchange.

Naming note: `build_solver()`'s parameter is `use_planning`, not
`planning` — mirrors the earlier `use_retrieval`/`retrieval()` pattern,
avoiding a parameter shadowing the module-level solver function of the
same name inside the function body (hit this exact issue once already
while writing it, first draft used `globals()["planning"]()` as an ugly
workaround before renaming the parameter instead).

**Verified without any API calls:** `elicit(suite="humaneval",
tool_use=True, planning=True)` builds correctly; solver step counts are
right for every toggle combination tested (3 steps for `planning` alone:
system message, planning turn, generate; 7 steps with every toggle on).

**Not yet done, per Phase 7's own criteria — needs real model calls, and
per the phase's stated design should run on humaneval specifically:**
the n=20 smoke test (confirm plans genuinely differ per question, confirm
the final answer references the plan, not just that the code runs), and
the full ablation once Phase 5's own humaneval/Docker smoke test has
independently confirmed that pipeline works — planning's empirical
validation depends on that suite already being validated, not just on
API budget being free again.

---

## Scope decision — 4-component Shapley study on Qwen + Llama, and why

Reconsidered project scope mid-session: rather than stopping at "looks
complete" (roughly Phase 4/5), decided to push toward a real Shapley
attribution study (Phase 10) plus cross-model comparison (a scoped
version of Phase 12) on Together-hosted open-weight models specifically
— Qwen and Llama, tying back to the earlier discussion about aligning
this project with the industry shift toward self-hosted/open-weight
inference. This is a substantially bigger undertaking than the earlier
"minimum to show someone" framing — most of the actual headline
deliverable, not a light next step — so scoped it deliberately rather
than just diving in.

**Retrieval dropped from the component set.** Decided to do 5 (later 4,
see below) of the README's 6 components rather than all 6, and cut
retrieval specifically — it was the most troubled component all session
(three real bugs: a random-Wikipedia corpus that returned near-random
passages on real GPQA/MATH questions, a Wikipedia rate limit, and a
MediaWiki API quirk that silently discarded 95% of fetched articles), and
even after the v2 category-filtered rebuild finished, its relevance was
never smoke-tested. Implementation: commented out (not deleted) in
`elicit_task.py` — the `retrieval()` solver, its BM25 helpers
(`_load_bm25_index`, `_retrieve`), the now-unused `json`/`lru_cache`
imports, and the two toggle call-sites in `build_solver()`/`elicit()`.
`use_retrieval`/`retrieval` parameters are left in place but now
harmless no-ops (verified: passing `retrieval=True` builds a `Task`
successfully and simply excludes the retrieval step) — kept this way
deliberately so re-enabling later is a matter of uncommenting, not
rewiring. A large comment block at the top of the commented section
explains the decision and exactly what to uncomment.

**The v2 corpus did finish before being shelved, worth recording for the
record even though it's not being used right now:** 3916 docs (down from
v1's 5000, since not every category-collected title had a usable
extract). Contamination check flagged 4 "substantive" hits this time
(v1 had zero) — read all 4 by hand: 3 were the phrase "mathbb r to
mathbb r be a function" (generic function-definition boilerplate, "let
f: R→R be a function," matching a genuine physics article on the
Ermakov–Lewis invariant) and 1 was "there is no limit on the number of"
(a coincidental generic English phrase matching an unrelated academy
article). Same false-alarm pattern as v1's all-digit-shingle hits —
confirmed clean, not real contamination, but never got to the actual
relevance spot-check before the scope decision to drop it.

**Real pricing pulled live before estimating anything**, not from
memory: `openai/gpt-4o-mini` $0.15/$0.60 per 1M in/out (web search,
confirmed against multiple sources), `together/Qwen/Qwen2.5-7B-Instruct-
Turbo` $0.30/$0.30, `together/meta-llama/Llama-3.1-8B-Instruct-Turbo`
$0.18/$0.18 (both via Together's own `/v1/models` API — same method used
earlier for the Qwen3.5 pricing check). Picked Llama-3.1-8B specifically
as the size-matched pairing with Qwen's 7B, to avoid confounding "model
family" with "model size" on top of the already-flagged cross-family (vs.
Phase 12's originally-designed same-family) caveat.

**Cost model: real anchors + a flagged estimate for untested components.**
Qwen2.5-7B's own real MATH data gave two genuine anchors: `cot` alone
measured 174 input / 829 output tokens/sample; `cot+critique` measured
3336 input / 1725 output tokens/sample — a real, measured jump showing
that each additional turn re-sends the *entire* prior conversation as
input, so cost compounds with stacked components, not just adds linearly.
Built a small turn-based cost model calibrated to that real pattern
(830 output tokens/turn, ~200 tokens fixed prompt overhead per turn) to
estimate the 4 components that have never actually been run
(`tool_use`, `planning`, `best_of_n`, `multi_critic`) — explicitly
flagged as an estimate, not a measurement, with `planning` the most
trustworthy of the four (its structure — one bounded extra turn — is
known exactly from the code) and `tool_use`/`multi_critic` the least
(a real tool round-trip could differ from the 1-turn assumption).

**Decision: drop to 4 components (cut `best_of_n`), keep 5 seeds, keep
Qwen2.5-7B/Llama-3.1-8B (don't go smaller).** Full reasoning and the
actual cost/power numbers behind each call:

- *Components, 5→4:* dropping any one component halves the sweep
  (2⁵=32 → 2⁴=16 configs) regardless of which — so the choice was about
  which component to cut, not whether cutting helps. Cut `best_of_n` for
  two independent reasons that both point the same way: (1) its
  headline property — trading off against budget — can't be
  demonstrated without Phase 11 (budget axis), which isn't in scope
  right now, so its most distinguishing feature is structurally
  unreachable in this study anyway; (2) unlike `planning` (already
  built) and `critique` (already built and validated), `best_of_n` is
  the only remaining component with zero code written, needing real
  new design work (an N-per-budget definition, a diversity check) before
  it's even buildable. `multi_critic` (also unbuilt) was kept over it
  specifically because it synergizes with the cross-model plan already
  in scope — the same-model-vs-different-model critic comparison its own
  acceptance criteria require becomes cheap once two models already
  exist for the cross-model piece.
  Cost at 4 components (`critique`, `tool_use`, `planning`,
  `multi_critic`), GPQA only, Qwen+Llama combined:
  **$45.58 at 5 seeds, $27.35 at 3 seeds** — down from 5 components'
  $119.45 / $71.67.

- *Seeds, 5 vs 3:* tested this against the project's OWN real result
  rather than a generic power-analysis explanation. GPQA's bare-vs-cot
  finding (the one clean significant result so far) was p=0.0401 at 5
  seeds (275 pooled disagreements, 120/155 split) — barely under 0.05.
  Recomputed the same win/loss ratio scaled to what 3/5 the seeds would
  likely have produced (165 disagreements, ~72/93 split):
  **p=0.1192 — would NOT have been significant.** This is the concrete
  cost of fewer seeds, illustrated on a result this project already has,
  not a hypothetical: the finding you already found would probably have
  been missed. Kept 5 seeds.

- *Model size:* checked smaller options live rather than assuming —
  `Llama-3.2-1B-Instruct` is real and cheap ($0.06/$0.06), but
  `Qwen2.5-3B-Instruct`/`1.5B-Instruct` showed `$0/$0` pricing on
  Together, which more likely means they're not available as plain
  pay-per-token serverless endpoints than that they're actually free —
  flagged as unverified rather than trusted at face value. More
  importantly: going smaller reopens the exact headroom-validation risk
  already paid for once this session for GPQA/MATH (a too-weak model
  can land near-floor the same way a too-strong one lands at ceiling,
  and you don't know which without a fresh n=20-40 pilot per model) —
  and the earlier Qwen3.5-9B lesson (a "smaller"/newer model that turned
  out extremely verbose and expensive to get clean output from) is
  direct evidence that model size alone doesn't predict cost or
  reliability. Kept Qwen2.5-7B / Llama-3.1-8B.

**Final scoped estimate: ~$27-46 for the 4-component Shapley sweep on
GPQA, Qwen + Llama, 5 seeds** (upper end at 5 seeds, lower end would be
3 seeds — kept 5 per the power argument above, so **$45.58** is the
number actually being planned against). This is for GPQA alone — no
budget axis, no second suite. Time is probably the tighter constraint
than dollars given this session's actual experience with rate limits and
interruptions on a much smaller sweep; planning for this to span
multiple sessions of real wall-clock time, not one sitting.

**Not yet done:** none of `tool_use`, `planning`, `best_of_n`, or
`multi_critic`'s token-cost assumptions have been checked against a real
pilot. Recommended next step (not yet executed): a cheap real pilot
(n=20-40, 1 seed, single-component-on configs only) specifically to
replace the estimated turn-costs with measured ones before committing
the full ~$46 / 16-config / 5-seed / 2-model sweep.

---

## Phase 9 — `multi_critic` component built, reusing Inspect's own cross-model support

Built while the gpt-4o-mini MATH ablation ran in the background (pure
code, no API calls, same reasoning as the earlier Phase 5/6/7 work).

Checked Inspect's own `self_critique()` source before writing anything
custom — it already accepts an alternate `model=` parameter specifically
for cross-model critique ("you don't need to use the model being
evaluated"), so `multi_critic` didn't need a hand-rolled solver like
`planning`/`retrieval` did. It's a thin, documented wrapper:
`multi_critic(critic_model: str | None = None)` → `self_critique(model=
critic_model)`. `critic_model=None` means same-model critique
(behaviorally identical to the existing `critique` toggle) — deliberate,
not a gap: Phase 9's own acceptance criteria require comparing
same-model-critic vs different-model-critic behavior, and having both be
the same function with a different argument is what makes that a clean,
directly-comparable check rather than two separate code paths. For the
real ablation sweep, `critic_model` gets set to "the other model in the
pair," so `multi_critic` is a genuine cross-model check there, not an
accidental duplicate of `critique`.

Verified without any API calls: builds correctly for both same-model and
cross-model configs, correct solver step counts (3 for `multi_critic`
alone, 7 with cot/tool_use/planning/critique/multi_critic all on).

Noted but deliberately NOT touched: `run_ablation.py`'s `MODEL`/`SUITE`/
`CONFIGS` still reflect the original 2-component (cot/critique) pairwise
ablation, because that script is the one currently running the live
Phase 4 gpt-4o-mini MATH-vs-GPQA comparison in the background. Editing it
now risks corrupting a future restart of that in-flight work (checkpoint
resume depends on the file staying as-is). The 16-config Shapley sweep
needs its own script — not yet written.

## Phase 12 — Second model selection: Llama-3.1-8B inaccessible, replaced with gemma-3n-E4B-it

Attempted the planned GPQA headroom pilot for the locked-in model pair
(Qwen2.5-7B + Llama-3.1-8B-Instruct-Turbo) — this suite-specific check
was itself overdue: Qwen2.5-7B had only ever been validated on MATH this
session, never GPQA, which is the suite the actual study targets.

**Qwen2.5-7B on GPQA: clean.** n=40, bare, accuracy=0.350 — good
headroom, consistent with gpt-4o-mini's GPQA range all session, no
ceiling/floor concern.

**Llama-3.1-8B-Instruct-Turbo: inaccessible.** `NotFoundError` (404,
`model_not_available`) on the very first call, despite showing valid
pricing in Together's `/v1/models` listing when checked earlier —
confirmed here that pricing-list presence does NOT mean inference
access; these are two different checks and only one had been done.
Tried every other Llama size/variant available on the account (8B in
multiple naming forms, 70B in two forms, 3B, 1B) — **only
`Llama-3.3-70B-Instruct-Turbo` actually works for inference.** No
distinguishing field in the API response explained the pattern; this
looks like an account-level serverless-tier restriction on Together's
side, not something fixable in code.

**Reopened the same-family-vs-cross-scale tradeoff this design was
built to avoid** — using the 70B model would confound family and scale
again, the exact problem the original small-pair decision was meant to
prevent. Screened alternatives from the account's actual available-model
list instead of assuming Llama was the only option, since the user
confirmed company/brand doesn't matter, only behavior:

| model | n | accuracy | avg out tok/sample | no-answer rate |
|---|---|---|---|---|
| `gpt-oss-20b` | 30 | 0.533 | 1133 | **9/30 (30%)** |
| `gemma-3n-E4B-it` | 30 | 0.400 | 5 | 0/30 |

(`LFM2.5-8B-A1B` and `gemma-4-31B-it` were screened out earlier at n=2 on
output-token volume alone — 1925 and 1417 tokens/sample respectively,
both in the range that had flagged trouble before — and not pursued
further.)

`gpt-oss-20b`'s higher raw accuracy is misleading: **30% of bare-condition
samples never produced a parseable answer** — the identical failure
signature already seen and rejected once this session with Qwen3.5-9B (a
reasoning-heavy model that burns output budget without reliably
terminating in the expected format). Ruled out despite the better
headline number, same reasoning as before: an unreliable answer format
corrupts the measurement more than a few points of accuracy matters.

`gemma-3n-E4B-it` was clean on every axis: real headroom (0.400, matches
the range other models have landed in on GPQA all session), essentially
zero verbosity (5 output tokens/sample in the bare condition — the model
just states a letter), and **zero** answer-format failures across 30
samples. Also cheaper than the original Llama-3.1-8B plan ($0.06/$0.12
vs $0.18/$0.18 per 1M tokens), so the existing ~$45.58 cost estimate
goes down, not up, with this substitution.

**Locked in: Qwen2.5-7B-Instruct-Turbo + `together/google/gemma-3n-E4B-
it`.** Both validated on GPQA specifically now, not carried over from
MATH validation. README and phases.md updated to replace the stale
Llama-3.1-8B references (Findings section placeholder, Phase 12,
Phase 3's feasibility-estimate note).

## `tool_use` on GPQA crashed the whole task — GPQA had no sandbox

Found while starting the token-cost pilot for the 4 in-scope components,
before even getting to real cost numbers. `elicit(suite="gpqa",
tool_use=True)` doesn't just run tool_use ineffectively on GPQA (the
outcome Phase 5's own requirements already warned about, "little for a
code tool to do") — it **crashes the entire task** the moment the model
actually invokes the tool: `ProcessLookupError: No sandbox environment
has been provided`. Only `humaneval`'s `TaskAdapter` has `sandbox=
"docker"` set; `gpqa` has none, and Inspect's `python()` tool needs a
real sandbox to execute in regardless of which suite is asking.

This is a real conflict between two decisions that were never checked
against each other: the "GPQA only for now" scope decision, and `tool_use`
being one of the 4 locked-in components. Locking in GPQA-only implicitly
assumed every component would at least *run* there, which was never
verified for `tool_use` specifically until now.

**Fix:** `elicit()` now sets `sandbox = adapter.sandbox or ("docker" if
tool_use else None)` — only spins up Docker when `tool_use` is actually
toggled on for that specific config, not unconditionally for every GPQA
run. This matters because 15 of the 16 configs in the sweep don't use
`tool_use` at all and shouldn't pay Docker's per-sample startup cost.

Verified the fix: same n=10 Qwen2.5-7B `tool_use=True` GPQA pilot that
crashed before now completes cleanly (`status: success`, accuracy=0.3,
1 real tool call across 10 samples — confirms the model does
occasionally reach for the tool on GPQA questions, not that it's
silently unused). One-time Docker image pull for Inspect's sandbox
tooling (`aisiuk/inspect-tool-support`, ~700MB) happened on this first
run; cached for subsequent ones.

**Standing caveat, not fixed by this, just no longer fatal:** Phase 5's
own concern about `tool_use` having "little for a code tool to do" on
non-code multiple-choice questions is still a live, separate question —
fixing the crash means `tool_use` on GPQA can now produce a *real*
number, but that number might legitimately be small/null just because
GPQA rarely calls for a calculator. That would be a valid finding, not a
bug, and is exactly what the pilot below is partly for.

## Token-cost pilot (n=25, GPQA, both locked-in models) — found and fixed a second real bug, then got real cost numbers

Ran bare + each of the 4 in-scope components alone, on both Qwen2.5-7B
and gemma-3n-E4B-it, specifically to replace the earlier turn-based cost
*estimate* with measured numbers.

**Found a second real bug before getting to the cost question at all:
`planning` was structurally broken, not just weak.** Raw pilot result —
Qwen2.5-7B bare 0.480 → planning 0.040; gemma-3n-E4B-it bare 0.400 →
planning 0.120. Too large a collapse to be a real effect (stderr ~0.10);
checked transcripts before believing it. Root cause: `planning()`
appended the plan-request, got the plan back as a real assistant turn,
then just returned — relying on `build_solver()`'s own trailing
`generate()` call to produce the final answer. But with the conversation
ending on the model's OWN plan turn and no explicit next instruction,
the model had nothing telling it to stop planning and start answering,
so it just repeated/continued the plan. Confirmed directly in
transcripts: EVERY sample's final "answer" was the plan text verbatim,
scorer correctly reading `(no ANSWER: line found)` on all of them.

**Fix:** added `PLANNING_FOLLOWUP_PROMPT`, an explicit user message
appended after the plan turn telling the model to now actually answer,
using the plan, following the system prompt's format instructions
(deliberately doesn't hardcode "ANSWER: X" or `\boxed{}` — `planning()`
is meant to be suite-agnostic, so it defers to whatever format the
active suite's own system prompt already specifies). Mirrors how
Inspect's own `self_critique()` re-poses the question and explicitly
asks for a new answer in its `completion_template`, rather than trusting
the model to infer that on its own — same lesson, independently
rediscovered.

**Verified the fix, same exact configs:**

| model | before fix | after fix | no-answer rate |
|---|---|---|---|
| Qwen2.5-7B | 0.040 | **0.320** | 0/25 (was 25/25) |
| gemma-3n-E4B-it | 0.120 | **0.360** | 1/25 (was 25/25) |

Both still land somewhat below their bare baselines (0.480, 0.400) —
that residual gap might be a real, modest planning-hurts-a-little effect
worth reporting later, or might narrow further with more samples; no
longer a broken pipeline either way, which is what mattered here.

**Full pilot results, all 5 configs, real per-sample token costs (n=25,
GPQA, temperature=0):**

| model | config | accuracy | in/sample | out/sample |
|---|---|---|---|---|
| Qwen2.5-7B | bare | 0.480 | 254 | 5 |
| Qwen2.5-7B | critique | 0.440 | 935 | 5 |
| Qwen2.5-7B | tool_use | 0.360 | 538 | 5 |
| Qwen2.5-7B | planning | 0.320 | 709 | 101 |
| Qwen2.5-7B | multi_critic | 0.440 | 909 | 5 |
| gemma-3n-E4B-it | bare | 0.400 | 241 | 5 |
| gemma-3n-E4B-it | critique | 0.400 | 871 | 115 |
| gemma-3n-E4B-it | tool_use | 0.360 | 270 | 5 |
| gemma-3n-E4B-it | planning | 0.360 | 708 | 677 |
| gemma-3n-E4B-it | multi_critic | 0.400 | 875 | 118 |

Side observation: both models answer GPQA in ~5 output tokens/sample in
the bare condition (just a letter, as instructed) — GPQA's answer format
is far more concise than MATH's free-form answers, which is exactly why
the earlier turn-based cost model (calibrated on Qwen's MATH data,
830 output tokens/turn) was a significant overestimate for this suite.

**Revised cost estimate using these real numbers** (additive
approximation — sum each active component's measured overhead over bare;
likely a floor, not a ceiling, since stacking multiple turn-adding
components probably compounds context-resend cost the way `cot+critique`
did on MATH earlier, not just adds):

- Qwen2.5-7B, 16 configs × 5 seeds × n=198: **$6.39**
- gemma-3n-E4B-it, 16 configs × 5 seeds × n=198: **$1.93**
- **Total: ~$8.31** — down from the earlier $45.58 estimate, because
  that estimate was built on MATH-shaped token costs and GPQA is much
  cheaper per sample. Even with a generous 2-3x margin for compounding
  effects the additive model doesn't capture, this is comfortably
  affordable.

---

## Open item / next entry to add

- [x] Record pooled McNemar p-value: bare vs cot — p=0.0401, significant
- [x] Record pooled McNemar p-value: cot vs cot+critique — p=0.887, not
      significant
- [x] Write one paragraph interpreting both in plain language — done
      above
- [x] Fix MATH headroom — switched to `intermediate_algebra` (0.552 at
      n=500), documented above
- [x] Spot-check intermediate_algebra for both false negatives (113
      candidates read, 0 found) and false positives (54 read, 0 found) —
      not the literal random-sample protocol `spot_check.py` was built
      for, but a strictly more targeted version of the same check
- [ ] GPQA still has no false-positive check at all (only the
      system-prompt bug and the CoT-leak fix have been examined) — run
      `spot_check.py` against a GPQA log before treating GPQA's numbers
      as equally scrutinized
- [x] Run the full 4-config × 5-seed ablation on intermediate_algebra —
      DONE, all 20 runs completed cleanly on `together/Qwen/Qwen2.5-7B-
      Instruct-Turbo`, saved to `results/ablation_summary_math.json`
- [x] Pooled McNemar on MATH: bare vs cot — p=0.0001, significant, but
      **cot is worse than bare** (opposite direction from GPQA)
- [x] Pooled McNemar on MATH: cot vs cot+critique — p=0.0001, significant,
      critique makes it worse again. Both documented above, with the
      confound (suite change AND model change at once) stated explicitly
      — this is NOT yet a confirmed cross-suite finding
- [x] Fix `MATH_SYSTEM` to have a true bare-vs-cot split — DONE, see
      "MATH_SYSTEM prompt-leak fix" entry above. Confirmed a large effect
      (leaky bare was ~0.552, properly-bare is ~0.25) — every MATH result
      before this fix is provisional, including the Qwen2.5-7B ablation
      and the 7-subject headroom comparison (rankings probably still
      hold, absolute numbers don't)
- [ ] Re-run the MATH ablation on `openai/gpt-4o-mini` with the fixed
      prompt — IN PROGRESS (5th launch attempt after another sleep
      interruption; checkpointing means each interruption now only costs
      the seed in flight, not everything before it — see "gpt-4o-mini
      MATH ablation" entry above). Checkpoint as of last check: `bare`
      seeds 1-2 done (0.244, 0.230); not yet complete.
- [ ] GPQA still has no false-positive check at all (only the
      system-prompt bug and the CoT-leak fix have been examined) — run
      `spot_check.py` against a GPQA log before treating GPQA's numbers
      as equally scrutinized
- [ ] Held-out/private contamination slice — still not done for either
      suite
- [ ] Power/sample-size justification and compute/dollar feasibility
      estimate for the eventual 64-config sweep — still not done, though
      real per-sample cost data now exists for both gpt-4o-mini (n=500
      intermediate_algebra: 611,833 tokens, 4m42s) and Qwen2.5-7B-
      Instruct-Turbo (n=40: 39,100 tokens, 23s) to base it on
- [x] Phase 5 code implementation — DONE: `tool_use` toggle,
      `humaneval` adapter (dataset ID fixed, `code_execution_match`
      actually implemented), Docker confirmed working. See "tool use +
      HumanEval" entry above.
- [ ] Phase 5 empirical steps — NOT DONE: n=20 smoke test (sandbox
      spin-up, real tool invocation, scorer correctness), full 16-config
      ablation on humaneval, pooled McNemar, tool-invocation-rate spot
      check. Blocked on the gpt-4o-mini MATH ablation finishing first
      (both need OpenAI API budget, avoiding concurrent pressure after
      the rate-limit incident above).
- [x] Retrieval — DROPPED FROM SCOPE (not deleted). v2 corpus finished
      (3916 docs, contamination check clean after manual review of 4
      flagged hits — all generic-phrase false alarms, same pattern as
      v1). But cut from the component set before the relevance
      spot-check ever happened — see "Scope decision" entry above for
      why (most troubled component all session, and the study is doing
      4-5 of 6 components rather than all 6). Code commented out in
      `elicit_task.py`, not removed; `retrieval`/`use_retrieval` params
      are now harmless no-ops. Easy to re-enable later if there's time.
- [x] Phase 7 (`planning`) toggle code — DONE: a genuine separate
      generation turn (not relabeled `chain_of_thought()`), verified
      wiring across all toggle combinations. See "planning component"
      entry above.
- [ ] Phase 7 empirical steps — NOT DONE: needs real model calls, and per
      its own criteria should run on humaneval — blocked on Phase 5's own
      smoke test confirming that suite's Docker/sandbox pipeline first,
      not just on API budget freeing up.
- [x] Scope decided for the Shapley + cross-model study — see "Scope
      decision" entry above: 4 components (`critique`, `tool_use`,
      `planning`, `multi_critic` — `best_of_n` and `retrieval` both
      cut), 5 seeds, GPQA only for now.
- [x] Second model locked in: Qwen2.5-7B-Instruct-Turbo +
      `together/google/gemma-3n-E4B-it` (NOT Llama-3.1-8B — inaccessible
      on this Together account, see "Second model selection" entry
      above). Both validated on GPQA specifically (n=30-40). Estimated
      cost ~$45.58 or lower (gemma-3n-E4B-it is cheaper than the
      original Llama-3.1-8B pricing assumption).
- [ ] `best_of_n` toggle — NOT BUILT, deliberately deferred (needs the
      budget axis to show its actual value, per the scope decision above)
- [x] `multi_critic` toggle — BUILT (thin wrapper around Inspect's own
      `self_critique(model=...)`, see "Phase 9" entry above). All 4
      in-scope components now exist in `elicit_task.py`.
- [x] Pilot the 4 real components at small scale (n=25, GPQA, both
      locked-in models) — DONE, real per-component token costs measured.
      Found and fixed two real bugs along the way, both more important
      than the cost numbers themselves: (1) `tool_use` crashed the whole
      task on GPQA (no sandbox configured — fixed, `elicit()` now adds
      Docker only when `tool_use` is actually toggled on); (2)
      `planning` was silently broken (0.04-0.12 accuracy, 25/25
      no-answer — missing a follow-up instruction telling the model to
      actually answer after planning, not a real reasoning collapse —
      fixed, recovered to 0.32-0.36). See both entries above.
- [x] Revised cost estimate using real (not estimated) per-component
      costs: **~$8.31** for the full 16-config × 5-seed × 2-model sweep
      on GPQA — down from the earlier $45.58 estimate, since that one was
      calibrated on MATH's much longer free-form answers and GPQA turns
      out to answer in ~5 output tokens/sample. Comfortably affordable
      even with margin for compounding effects the additive estimate
      doesn't capture.
- [x] A dedicated sweep-runner script for the 16-config × 5-seed ×
      2-model Shapley study — DONE: `run_shapley_sweep.py`, deliberately
      separate from `run_ablation.py` per the reasoning above. Reuses the
      same checkpoint-after-every-seed / resume-from-partial pattern
      `run_ablation.py` already proved out. See "Phase 10" entry below.
- [x] Full 4-component (2⁴=16 config) × 5-seed Shapley sweep on GPQA,
      model 1 of 2 (`Qwen2.5-7B-Instruct-Turbo`) — DONE, all 80 runs
      clean. Model 2 (`gemma-3n-E4B-it`) IN PROGRESS after a real
      20-hour API-timeout stall, investigated and resumed cleanly. See
      "Phase 10" entry below for both.

---

## Phase 10 — Shapley sweep: `run_shapley_sweep.py`, Qwen complete, Gemma stalled and resumed

Wrote `run_shapley_sweep.py` per the open item above: separate script
from `run_ablation.py` for the same corruption-risk reason stated in the
Phase 9 entry, reusing the exact same proven patterns (checkpoint after
every seed, resume-from-partial via a `(label, seed)` done-set loaded
from the existing summary JSON, `MAX_CONNECTIONS=5`) rather than
rediscovering them. `MODEL_PAIR` iterates the two locked-in models in
sequence, each writing its own `results/shapley_sweep_gpqa_{model_slug}.
json` — mirrors the earlier suite-specific-output-path fix that prevented
a MATH run from clobbering the GPQA one.

**Qwen2.5-7B-Instruct-Turbo: full sweep completed cleanly.** All 16
configs × 5 seeds = 80 runs, saved to `results/shapley_sweep_gpqa_
together-qwen-qwen2.5-7b-instruct-turbo.json`. No errors, no
interruptions during this model's run.

**gemma-3n-E4B-it: interrupted partway, twice.** First launch got through
only `bare` and `multi_critic` (2/16 combos, 5/5 seeds each) before the
background process was killed (external interruption, not a crash — no
error in the process's own output). Resumed via the same script (confirmed
the checkpoint-skip logic works exactly as designed: all 80 already-done
Qwen runs and gemma's 2 done combos were skipped, not re-run).

**Second interruption: a genuine ~20-hour API stall, not a transient
blip.** After resuming, `planning` finished cleanly (5/5 seeds), then
`planning+multi_critic` seed 1 got stuck retrying `APITimeoutError` on 5
specific samples (42, 45, 46, 47, 48) against `google/gemma-3n-E4B-it`,
continuously, from 13:03:52 to 08:40 the next day — 48+ consecutive retry
attempts at a 30-minute capped backoff, zero progress. Confirmed via
`inspect_ai`'s own `GenerateConfig` that `max_retries` defaults to `None`
(unbounded) — left alone, this would have retried forever rather than
eventually failing out on its own.

Investigated before just restarting blind, since 20 hours of consistent
failure is long enough to suspect a real, recurring problem rather than
bad luck:
- Checked the 5 stuck samples' question length against the full GPQA
  Diamond distribution (mean 431 chars, stdev 261) — all 5 landed at
  116-613 chars, unremarkable, no outlier. Ruled out "these specific
  questions are pathologically long" as the cause.
- Noted that exactly 5 samples stuck matches `MAX_CONNECTIONS=5` exactly
  — consistent with these just being the last batch still in-flight when
  the API started timing out, not specially cursed content.
- Direct live test call to `google/gemma-3n-E4B-it` (bypassing
  `run_shapley_sweep.py` entirely, via the OpenAI-compatible client
  against Together's endpoint) succeeded in 0.88s. Confirms the API was
  healthy again at investigation time — the stall was a real but
  apparently transient (if very long-lived) Together-side reliability
  issue specific to that model, most likely aggravated by
  `planning+multi_critic` being a heavier multi-turn request than `bare`.

**Killed the stuck process and relaunched the same way** (same script,
same checkpoint-resume). Confirmed clean this time: `planning+
multi_critic` finished all 5 seeds with sane accuracy values (0.268,
0.318, 0.298, 0.318, 0.298 — consistent with the other combos, no
lingering damage from the stall), then continued cleanly through
`tool_use` and `tool_use+multi_critic` (5/5 seeds each) with zero
`APITimeoutError` occurrences in the new run's log.

**Status as of last check: 6/16 gemma combos fully done** (`bare`,
`multi_critic`, `planning`, `planning+multi_critic`, `tool_use`,
`tool_use+multi_critic`), `tool_use+planning` in progress (1/5 seeds),
process confirmed still running. 9 combos remain after the current one.
Not yet done: the final `_save_summary()` call only happens once a model's
full loop completes, so `results/shapley_sweep_gpqa_together-google-
gemma-3n-e4b-it.json` won't reach 16/16 until the whole gemma sweep
finishes — and Phase 10's own follow-up (Shapley attribution + interaction
term computation over both models' summary JSONs, referenced in
`run_shapley_sweep.py`'s own closing print statement) hasn't been started.

**Standing note for later:** `run_shapley_sweep.py` still has no
`max_retries` or per-attempt `timeout` cap (same as `run_ablation.py`
before it). Given this stall lasted 20 hours specifically because retries
are unbounded, consider adding an explicit cap before the next long
unattended run — not fixed now, since restarting mid-sweep-design-change
would have meant validating a second thing at once instead of just
confirming the resume worked.

**Sweep finished — then one more bad cell surfaced on the last check.**
The gemma run went on to complete all 16 combos cleanly (no further
`APITimeoutError` occurrences in the post-restart log), giving 16/16
combos × 5/5 seeds for both models... except `critique+planning+
multi_critic` seed=1 came back `accuracy=nan`, not a real number.

**Root cause: a different failure mode from the timeout stall — an
unhandled `BadRequestError`, not a retry loop.** `read_eval_log()` on that
seed's `.eval` file showed `status: error`, 164/198 samples completed
before the whole run aborted. The embedded error was Together returning
`400 - Input validation error` on one specific sample's request to
`google/gemma-3n-E4B-it`. `critique+planning+multi_critic` builds the
longest, most turn-heavy message sequence in the entire sweep (system →
question → plan request → plan → "now answer" → critique → "revise" →
...), and Together's request validation rejected one particular instance
of it. Inspect's default behavior aborts the whole eval on the first
unhandled sample exception rather than skipping just that sample, so one
bad request cost the entire seed, not just one data point.
`get_accuracy()`'s existing `except Exception: return nan` fallback (in
`run_shapley_sweep.py`) is why this surfaced as a silent `nan` in the
summary JSON instead of a visible crash — worth knowing, since it means
future runs could have the same failure mode without an obvious error at
the top level; only checking every accuracy value against `nan` (or
reading `log.status` directly) surfaces it.

**Reran just that one (config, seed) directly** — `elicit(suite="gpqa",
critique=True, planning=True, multi_critic=True, critic_model=
"together/Qwen/Qwen2.5-7B-Instruct-Turbo")`, `model="together/google/
gemma-3n-E4B-it"`, `seed=1`, same params as the sweep — and patched the
result into `results/shapley_sweep_gpqa_together-google-gemma-3n-e4b-
it.json` in place. **Second attempt succeeded cleanly**: `status=success`,
accuracy 0.263, in line with the combo's other 4 seeds (0.298-0.328). The
400 didn't recur, which points to a transient Together-side issue on that
specific request rather than a reproducible bug in how the solver
composes messages for this component combination — but flagging it as a
pattern to watch for if it happens again on a *different* sample, which
would point at a real structural issue instead.

**Both models' sweeps are now genuinely complete: 16/16 combos × 5/5
seeds each**, for `Qwen2.5-7B-Instruct-Turbo`
(`results/shapley_sweep_gpqa_together-qwen-qwen2.5-7b-instruct-turbo.json`)
and `gemma-3n-E4B-it` (`results/shapley_sweep_gpqa_together-google-
gemma-3n-e4b-it.json`).

## Phase 10 (cont.) — Shapley attribution + pairwise interactions: built and run

Built `shapley_attribution.py`. Reuses `TOGGLES`/`config_label()` from
`run_shapley_sweep.py` directly rather than re-deriving the
label<->component mapping — one source of truth, so the two scripts can't
silently drift apart.

**Why Shapley, not just the ablation deltas already in this log:** a raw
bare-vs-single-component delta only measures a component's effect in one
context (nothing else on). Whether critique helps depends on what else is
active — the "critique adds nothing on top of cot, sits marginally below"
pattern found by hand earlier (GSM8K, single-seed GPQA, pooled GPQA) is
exactly a context-dependence effect. Shapley values average a
component's marginal contribution over every possible context. With all
2^4=16 coalitions actually measured (not sampled), the Shapley values and
pairwise Grabisch-Roubens interaction indices computed here are exact,
not estimated.

**Uncertainty via bootstrap, not point estimates alone:** each of the 16
coalitions has 5 real seeds. Resampled each coalition's 5 seeds with
replacement, recomputed every Shapley value + interaction term, 5000
iterations, took the 95% percentile CI. Efficiency property (Shapley
values must sum to `v(full) - v(bare)`) verified exactly on the point
estimate for both models before trusting anything downstream.

**Headline results (GPQA, letter_match accuracy, all deltas in
percentage points):**

| component | Qwen2.5-7B Shapley | gemma-3n-E4B-it Shapley |
|---|---|---|
| critique | +0.22 (CI includes 0) | +0.08 (CI includes 0) |
| tool_use | +0.08 (CI includes 0) | +0.55 (CI includes 0) |
| planning | +0.37 (CI includes 0) | **-1.32 (CI excludes 0)** |
| multi_critic | **-0.77 (CI excludes 0)** | **+1.20 (CI excludes 0)** |

**Only two main effects clear the 95% CI on either model, and they're on
*different* components in *opposite* directions** — `multi_critic` hurts
Qwen (-0.77pts) but helps gemma (+1.20pts); `planning` hurts gemma
(-1.32pts) but is noise for Qwen. Every pairwise interaction's CI
includes 0 for both models — no confirmed synergy or conflict between any
two components at this sample size, despite several point estimates
looking suggestive (e.g. Qwen's critique×multi_critic +0.57pts, gemma's
critique×multi_critic -1.14pts) — both consistent with zero given the CI
width.

**Reading this honestly: at n=198×5 seeds, only large single-component
effects are distinguishable from noise, and even those don't transfer
across models.** This is itself a real finding for the project's central
question (does scaffold-component attribution transfer across models) —
the answer on this data is "no, not even the sign transfers for the two
components that *are* individually significant" — but it's a
weaker-power result than the earlier pooled McNemar bare-vs-cot finding
(p=0.04 off 275 disagreements), because Shapley values here are built
from only 5 accuracy numbers per coalition rather than 198×5=990 paired
sample-level outcomes. A McNemar-style paired test at the sample level
(same idea as `mcnemar_test.py`, extended to the coalition-pair
comparisons the interaction terms care about) would have more power than
seed-level bootstrapping and is the natural next tightening if these
numbers need to support a stronger claim than "no interaction survived at
this power."

Not yet done: same computation on a second suite (MATH/intermediate_algebra
never got its own Shapley sweep, only the earlier 4-config ablation) to
check whether the attribution pattern is suite-specific too, on top of
already being model-specific.

## Phase 10 (cont.) — sample-level paired significance test: two of three "significant" findings don't survive

Built `shapley_significance_test.py`, the tightening flagged as the
natural next step in the prior entry. `shapley_attribution.py`'s
bootstrap CI resampled only the 5 per-seed COALITION MEAN accuracies per
coalition -- 5 numbers, full stop. But every Shapley main effect and
pairwise interaction is just a fixed weighted linear combination of
coalition means, and each coalition mean is itself an average over 198
real per-sample outcomes on the IDENTICAL question set every other config
saw (`shuffle=False`, same discipline `mcnemar_test.py` already relies
on). So the identical linear combination can be computed PER (sample,
seed) pair instead of per coalition -- one real observation per (k, s)
instead of one per coalition, ~990 instead of 5. Averaging the ~990
values reproduces the exact same point estimate as the coalition-mean
version (hard-asserted in code, not eyeballed -- confirmed exactly equal
for all 4 main effects and all 6 interactions, both models). Ran both a
paired t-test and a Wilcoxon signed-rank test (non-parametric, since the
per-sample statistic is a discrete-ish weighted sum of 0/1 differences,
not obviously normal) against a null of 0.

**Result: only 1 of the 3 previously-"significant" findings survives.**

| effect | seed-bootstrap CI (prior entry) | sample-level test (n=990) |
|---|---|---|
| Qwen `multi_critic` (-0.77pts) | excluded 0 | t p=0.104, Wilcoxon p=0.279 -- **not significant** |
| gemma `planning` (-1.32pts) | excluded 0 | t p=0.152, Wilcoxon p=0.534 -- **not significant** |
| gemma `multi_critic` (+1.20pts) | excluded 0 | t p=0.021, Wilcoxon p=0.017 -- **still significant** |
| all 6 pairwise interactions (both models) | all included 0 | all still not significant |

**Why the flip, and which one to trust:** the seed-level bootstrap
resamples-with-replacement from only 5 real numbers per coalition --
small enough that the resulting CI is mechanically bounded by whatever
those 5 numbers happened to be (can't produce a value outside their
observed range), which understates true uncertainty when 5 points happen
to look tightly clustered by chance. The sample-level test uses the real
per-question paired structure instead, the same principle
`mcnemar_test.py` was built around from the start of this project, and is
the more trustworthy number of the two -- consistent with why pooled
McNemar (sample-level) was adopted over single-run comparisons back in
Phase 4. Gemma's `planning` result is the clearest case: t-test p=0.152
and Wilcoxon p=0.534 disagree by a lot, meaning the per-sample distribution
is skewed enough that the parametric test's normality assumption is
shaky -- when the more conservative nonparametric test also says "not
significant," that's not a borderline call.

**Standing takeaway, revised from the prior entry: exactly one main
effect survives rigorous testing on this data -- gemma's `multi_critic`
benefit (+1.20pts, p<0.05 on both tests).** Every other main effect and
every pairwise interaction, on both models, is statistically
indistinguishable from noise at n=198x5 seeds. This is a much narrower
claim than "two models show opposite significant effects on two
different components" -- worth stating precisely this way in any eventual
writeup rather than the looser version from the previous entry.

**Caveat, restated (same one `mcnemar_test.py`'s pooled mode has always
carried):** the 5 seeds share the same fixed 198-question set, so the 990
"observations" aren't fully independent draws -- treat these p-values as
an approximation, not textbook-exact, same as every other pooled test in
this project.

---

## Phase 11 setup — before a full MATH sweep: pilot + power calc, and a runaway `tool_use` process caught mid-flight

Given GPQA's sweep died under multiple-comparison correction, decided
against just re-running the same 16-combo x 5-seed design on MATH and
hoping for better luck. Root cause of the GPQA failure wasn't bad luck --
`n=198`/5 seeds was validated once, against one big effect (bare vs cot,
+7pts, see "Scope decision" entry's seed justification), then reused to
test ~20 much smaller Shapley effects it was never checked against.
Built two scripts to fix the process, not just swap suites:

- `math_pilot.py`: bare + each of the 4 single components (5 configs,
  not the full 16), n=40, 3 seeds, both locked-in models. Saves per-seed
  log paths, not just accuracy -- needed for real sample-level variance,
  same reasoning as `shapley_significance_test.py`.
- `power_analysis.py`: pulls real per-sample paired outcomes (via
  `mcnemar_test.py`'s `sample_outcomes()`) for each component vs bare,
  computes required N for 80% power at a Bonferroni-corrected alpha
  (default corrects for 20 tests, matching GPQA's design). Explicitly
  documented as a lower bound in its own docstring: it sizes the
  isolated component-vs-bare delta, which GPQA's own data showed is
  usually the LARGEST version of an effect (context-dependent, e.g.
  gemma's multi_critic delta ranged -0.51pts to +3.13pts depending on
  what else was on) -- the real Shapley main effect averages over more
  contexts and is often smaller, and pairwise interactions need more N
  still.

**Bare accuracy on the fixed-prompt MATH_SYSTEM, confirmed clean at
n=40:** Qwen2.5-7B landed at 0.200-0.250 across 3 seeds -- consistent
with the earlier gpt-4o-mini validation (~0.25) that confirmed the
prompt-leak fix, and confirms real headroom exists on this suite/model
pairing (not near-ceiling or near-floor).

**Runaway `tool_use` process found mid-pilot, unrelated to the power
question but a real operational bug.** User flagged high CPU usage while
the pilot's `tool_use` step was running. Investigated via `docker stats`/
`docker top` rather than guessing: found NOT idle leftover containers
(20 of those existed too, from the finished GPQA sweep, but sat at
~0.02% CPU -- harmless) but one specific container with a `python3`
process pinned at ~100% CPU continuously for **9 hours 11 minutes**,
tied to the pilot's currently-active `tool_use` sample. Root cause: the
model invoked the sandboxed `python()` tool with code that hit an
infinite loop, and `elicit_task.py`'s `use_tools(python())` call never
set a `timeout` -- confirmed via `inspect(python)` that the tool
signature has always accepted `timeout: int | None = None`, defaulting
to unbounded. This is the exact same class of gap HumanEval's own
`code_execution_match` scorer already guards against with its own 30s
timeout (see Phase 5's "tool use + HumanEval" entry) -- that guard was
written for one code-execution path and never extended to the other.

**Fix, in two parts:**
1. Operational: killed just the runaway process inside the container
   (`docker exec <id> kill -9 <pid>`, not the whole container, since it
   was still needed by the in-progress eval) -- confirmed CPU dropped to
   0% immediately after. The eval recovered on its own: that one sample
   scored as an error/incorrect, checkpointing meant nothing else was
   lost, and the pilot continued into the next seed normally. Also
   cleaned up the 20 idle leftover sandbox containers from the finished
   GPQA sweep (harmless but no longer needed) plus 6 unrelated old
   exited containers from other projects, after confirming with the user
   given the bulk/destructive nature of the cleanup.
2. Code: added `timeout=30` to `python()` in `build_solver()`
   (`elicit_task.py`), mirroring HumanEval's existing convention exactly.
   Matters more for the eventual full MATH sweep than it did for GPQA --
   GPQA's own `tool_use` pilot found the model invokes the tool rarely
   (1 real call across 10 samples, see "tool_use on GPQA crashed the
   whole task" entry), but MATH problems plausibly invite more
   computational tool use (verifying algebra, iterative solving), so the
   exposure to this bug is likely higher there, not lower -- good that it
   surfaced now, in a 40-sample pilot, rather than partway through a
   500-sample sweep run.

**Standing note:** this was caught because the user happened to notice
CPU usage, not because anything in the harness would have surfaced it on
its own. `run_shapley_sweep.py`/`math_pilot.py` have no wall-clock guard
at the eval level either -- a stuck sample without the new `timeout=30`
fix would have silently held up one of 5 concurrency slots indefinitely
with no error, no log line, nothing to notice except elevated CPU if
someone happened to look. Worth keeping in mind for any future
unattended multi-day run: fixed now for `tool_use` specifically, but the
general pattern (one bad sample silently absorbing a concurrency slot
forever) isn't structurally guarded against everywhere.

**Pilot finished cleanly on its own** (not killed) -- all 30 runs (5
configs x 3 seeds x 2 models) completed. One real problem found on
inspection: gemma's isolated `planning` config failed 3/3 (100%) seeds,
all on the same `BadRequestError 400 Input validation error` seen once
before on GPQA, but here hitting 3 *different* MATH questions at
different points in each 40-sample batch -- a much higher rate than
GPQA's 1-in-80. Likely cause: MATH's heavier LaTeX-dense content
combined with `planning()`'s two-consecutive-user-message structure
(question, then plan-request, no assistant turn between) -- that exact
structure worked fine on GPQA's simpler content. Qwen's `planning`
config on the same MATH questions succeeded 3/3, so this looks
gemma+MATH-content-specific. Not resolved -- moot for now given the
provider switch below.

---

## Switch to DeepInfra: neither original model exists there, real pricing comparison, and a Qwen3-32B reasoning-mode bug caught before it corrupted anything

User requested switching providers for cost (Together was too
expensive), with `DEEPINFRA_API_KEY` already added to `.env`.

**Checked before assuming anything would be a drop-in swap.** Neither
`Qwen2.5-7B-Instruct-Turbo` nor `google/gemma-3n-E4B-it` -- the exact
pair this entire study (GPQA sweep, Shapley/interaction analysis, MATH
pilot) was built around -- exists in DeepInfra's model catalog
(confirmed via their live `/v1/openai/models` endpoint, not assumed).
This is not a config change, it's a full model-selection redo, same
category of event as Phase 12's "Llama-3.1-8B inaccessible" pivot.

**Real pricing pulled live** (DeepInfra's OpenAI-compatible `/models`
endpoint doesn't expose pricing, so used their pricing page directly)
comparing candidates against current Together rates:

| model | Together (current) | DeepInfra candidates |
|---|---|---|
| Qwen (7B) | Qwen2.5-7B-Instruct-Turbo $0.30/$0.30 | no 7B Qwen2.5 exists at all. Qwen3-14B $0.12/$0.24 (bigger, cheaper), Qwen3-32B $0.08/$0.28 (much bigger, still cheap), Qwen2.5-72B-Instruct $0.36/$0.40 (bigger AND pricier -- not a win) |
| gemma (E4B) | gemma-3n-E4B-it $0.06/$0.12 | gemma-4-E4B-it $0.02/$0.10 (closest size match, genuinely cheaper), gemma-3-4b-it $0.05/$0.10 |

DeepInfra is genuinely cheaper, but only by accepting bigger models --
there's no small Qwen2.5 available there at any price. This directly
intersects the standing "maybe small models can't use these components"
question from the GPQA null-result discussion. **User chose to lean into
that: `Qwen/Qwen3-32B` + `google/gemma-3-27b-it`**, deliberately larger
than the original pair, still cheap on DeepInfra.

**Provider mechanics, confirmed via source not docs:** `inspect_ai`'s
`openai_compatible.py` has no dedicated `deepinfra.py` provider -- it's
used via the generic `openai-api/<service>/<model>` path, which
auto-derives `{SERVICE}_API_KEY` (so `DEEPINFRA_API_KEY` already matched
what the user set) but requires `{SERVICE}_BASE_URL` explicitly, no
hardcoded default the way Together's dedicated subclass has one. Added
`DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai` to `.env`
(confirmed live via DeepInfra's own docs, not memory).

**Bug 1: Qwen3-32B's default `max_tokens` (65536) exceeds its own
40960-token context window**, 400s on every call until capped
explicitly. Fixed: `MAX_TOKENS = 4096` in `run_shapley_sweep.py`,
applied uniformly to every model (not just the one that needed it --
never let generation config differ between compared configs).

**Bug 2, much more serious: Qwen3-32B has an internal "thinking" mode
that fires on hard questions regardless of the system prompt's "do not
explain your reasoning" instruction.** Initial 20-sample GPQA pilot
looked survivable on the surface (accuracy 0.450, `no_answer=0/20`) but
the aggregate hid two things: `avg_out_tok=1781` in the BARE condition
(gemma's same condition: 4 tokens) was the actual tell, and a hand-built
"no answer" check was silently wrong (checked
`score.answer in (None, '')`, but `letter_match`/`boxed_match` actually
write a sentinel string like `"(no ANSWER: line found)"` on failure, not
`None` -- so the check reported 0 failures when there were real ones).
Read 8 raw transcripts directly rather than trusting the aggregate:
- 2 of 8 samples burned the full 4096-token budget on hidden reasoning
  with a completely EMPTY visible completion -- real no-answer failures
  the buggy check missed entirely.
- 1 sample had reasoning leak directly into the visible text AND broke
  the required `ANSWER: X` format (`**Answer:** A` instead), so it would
  fail scoring even though a human could read the right answer off it.
- Reasoning blocks up to 16,319 chars on samples that DID produce a
  clean final answer -- real hidden cost/latency even when not visibly
  broken.
This is the same class of problem that got Qwen3.5-9B rejected earlier
in the project ("turned out to be unusable at a reasonable token
budget"), just less visible this time since it didn't show up as an
obviously-wrong headline accuracy number -- caught only because the
avg-output-token anomaly prompted a transcript read, the same discipline
that caught the original CoT-leak bug in Phase 3.

**Fix, confirmed via live smoke test (not docs -- DeepInfra's own API
docs page for this model don't mention it):**
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, the
standard vLLM/Qwen3 convention. On the same hard GPQA question: output
tokens 1977 -> 5, same correct answer. Confirmed harmless on
gemma-3-27b-it (extra fields ignored, verified with a direct call) so
applied to both models uniformly via `EXTRA_BODY` in
`run_shapley_sweep.py`, threaded into both scripts' `inspect_eval()`
calls alongside `MAX_TOKENS`.

**Re-ran the same 20-sample pilot with both fixes in place, both suites,
both models, with the no-answer check itself also fixed:**

| suite | model | accuracy | avg output tok | no-answer |
|---|---|---|---|---|
| GPQA | Qwen3-32B | 0.550 | 5 | 0/20 |
| GPQA | gemma-3-27b-it | 0.350 | 4 | 0/20 |
| MATH | Qwen3-32B | 0.500 | 548 | 2/20 |
| MATH | gemma-3-27b-it | 0.350 | 8 | 0/20 |

GPQA fully clean now (accuracy even improved, 0.450->0.550, consistent
with removing the format-violation failures). MATH still shows elevated
output (548 tok, vs gemma's 8) and a 10% no-answer rate for Qwen3-32B --
read both failures directly: one was a plain-text "neither" instead of
`\boxed{...}` (an ordinary instruction-following miss, not a bug), the
other was a genuine repetition loop (`"Let's now consider the sum of the
roots..."` repeated verbatim until hitting the token cap) -- a known
general LLM failure mode, not thinking-mode-specific. Judged acceptable
to proceed with, not a blocker, given it's an order of magnitude better
than the pre-fix failure profile and both remaining causes are mundane.

All four (suite, model) combinations show real headroom -- none near
their floor or ceiling.

**Status: setup complete, code updated
(`run_shapley_sweep.py`/`math_pilot.py` now target
`openai-api/deepinfra/Qwen/Qwen3-32B` and
`openai-api/deepinfra/google/gemma-3-27b-it`, with `MAX_TOKENS=4096` and
`EXTRA_BODY` fixes baked in). The Together-era MATH pilot data
(`results/math_pilot_together-*.json`) and the full GPQA sweep/Shapley
analysis remain valid as a separate prior chapter on the old model pair
-- not comparable to anything run on the new models, but not
invalidated either.

## Re-ran the MATH pilot on the new DeepInfra models: a third bug (`multi_critic`'s config never reached the critic model), fixed properly this time

Reran `math_pilot.py` on `Qwen3-32B` / `gemma-3-27b-it`. Both models'
`bare`/`critique`/`tool_use`/`planning` configs completed cleanly, in a
healthy 0.275-0.625 accuracy range, no errors -- including gemma's
`tool_use`, which took 2h12m on one seed (a real outlier vs. the usual
1-3 minutes) but completed successfully; checked `docker stats`
immediately after and found no runaway process this time, unlike the
earlier 9-hour incident -- the `timeout=30` fix from that entry is
holding.

**gemma's `multi_critic` config failed all 3/3 seeds**, same
`BadRequestError 400 -- max_tokens=65536 cannot be greater than
max_model_len=40960` already fixed once for Qwen3-32B as a PRIMARY
model. Root cause: `multi_critic(critic_model)` passes a bare model-name
string into `self_critique(model=critic_model)`, which resolves it via
its own internal `get_model(model)` call with NO config -- meaning the
critic model (Qwen3-32B, when it's playing critic for gemma) never
inherited the `max_tokens`/`extra_body` fix applied to the primary
model. The fix only ever touched the eval-level `inspect_eval()` call,
which has no visibility into a critic model instantiated separately
inside a solver.

**Fix, first attempt broke immediately in an instructive way.** First
instinct: change `multi_critic()` to build a configured `Model` eagerly
(`get_model(critic_model, config=critic_config)`) and pass that into
`self_critique(model=...)` instead of a bare string. This failed on the
very first call: `PrerequisiteError: No DEEPINFRA_API_KEY defined` --
because `elicit(...)` is evaluated as a plain function argument to
`inspect_eval(elicit(...), ...)`, so it runs to completion (including
any eager `get_model()` call inside it) BEFORE `inspect_eval()` itself
gets a chance to load `.env`. `self_critique()`'s own model resolution
already happens lazily, inside its `solve()` closure, at actual
generation time -- specifically to avoid this exact problem. The eager
fix broke the thing it was trying to fix.

**Real fix: made `multi_critic()` itself `@solver`-decorated and lazy,**
matching `self_critique()`'s own pattern -- `get_model(critic_model,
config=critic_config)` now happens inside `multi_critic()`'s own
`solve(state, generate)` closure, called only at actual generation time
(after `.env` is loaded), then delegates to `self_critique(model=critic)
(state, generate)` directly. Threaded `critic_config: GenerateConfig |
None` through `multi_critic()` -> `build_solver()` -> `elicit()`;
`run_shapley_sweep.py`/`math_pilot.py` now build
`GenerateConfig(max_tokens=MAX_TOKENS, extra_body=EXTRA_BODY)` and pass
it as `critic_config` whenever `multi_critic=True`, mirroring the
primary model's own config exactly.

Verified with a direct smoke test before touching the pilot data:
gemma-with-Qwen3-32B-critic on 5 MATH samples, `status=success`, critic
model usage 743 tokens (40 output) -- no overflow, and `enable_thinking:
false` confirmed carrying through to the critic too (40 output tokens
for a critique response, not the thousands a "thinking" pass would
burn).

**Patched the 3 failed seeds directly** (checkpoint-resume wouldn't have
retried them on its own -- a `nan`-valued `(config, seed)` entry still
counts as "already done" to that logic, same gap noted in the earlier
GPQA nan-patching entry). All 3 succeeded cleanly on rerun (0.400,
0.375, 0.400) -- both models' pilots are now genuinely complete, 5
configs x 3 seeds each, zero remaining gaps:

| config | Qwen3-32B | gemma-3-27b-it |
|---|---|---|
| bare | 0.575, 0.550, 0.475 | 0.350, 0.375, 0.375 |
| critique | 0.475, 0.325, 0.400 | 0.525, 0.550, 0.525 |
| tool_use | 0.400, 0.475, 0.325 | 0.325, 0.275, 0.275 |
| planning | 0.575, 0.550, 0.625 | 0.375, 0.400, 0.400 |
| multi_critic | 0.475, 0.550, 0.500 | 0.400, 0.375, 0.400 |

Next: run `power_analysis.py` on both files to get real required-N
numbers for the components that have never been measured on MATH before
(`tool_use`, `planning`, `multi_critic`) -- the actual purpose of this
pilot, now unblocked.

## Power analysis results, and launching the real MATH Shapley sweep

`power_analysis.py` on both pilot files, Bonferroni-corrected for 20
planned tests (matching GPQA's design): **7 of 8 tested main effects are
comfortably powered at the planned n=500 x 5-seed budget** (critique:
N=154/93 needed vs 2500 available; tool_use: N=281/945; planning:
N=1491/1793, right at the edge but clears it; gemma's multi_critic:
N=587). The one gap: **Qwen's multi_critic effect (-2.5pts) needs
N=7416, about 3x the planned budget** -- a real, small effect that this
sweep won't have power to confirm, flagged going in rather than
discovered as a surprise later. Sharp contrast with GPQA, where nothing
survived correction at all -- direct evidence the earlier model upsize
(7B/E4B -> 32B/27B) fixed the actual problem, not just changed the
suite.

Updated `run_shapley_sweep.py`: `SUITE = "math"`, `LIMIT = 500` (was
still pointed at the old `gpqa`/`198` config from before the DeepInfra
detour). Launched the real 16-combo x 5-seed x 2-model sweep.

## Why the sweep would have taken ~12 days, and getting it to ~3

First estimate, from real pilot timing data (30 runs at n=40 totaled 4.49
hours) scaled by the sample-work ratio to the full sweep (160 runs x
n=500 vs 30 runs x n=40 = 66.7x): **~300 hours, ~12.5 days**, sequential.
Killed the sweep immediately (it hadn't even finished its first
n=500 run yet -- essentially zero sunk cost) rather than let a
12-day estimate ride.

**Lever 1: `MAX_CONNECTIONS` was still 5, calibrated against an OpenAI
rate-limit incident that has nothing to do with DeepInfra.** Never
re-tested. Ran a live 3-way comparison (n=40, same model/config/seed,
varying only `max_connections`): 5 -> 142.8s, 20 -> 94.0s (~1.5x
faster), 40 -> 103.1s (no further gain -- likely DeepInfra's own
throughput ceiling, not the client-side setting). Set
`MAX_CONNECTIONS = 20`.

**Lever 2: running both models in parallel.** First attempt used
`asyncio.gather()` over two `eval_async()` calls in one process --
failed immediately with `RuntimeError("Multiple concurrent calls to
eval_async are not allowed.")`, an explicit Inspect restriction, not a
provider issue. Fixed by using two separate OS processes instead (added
an optional `sys.argv[1]` model-index selector to
`run_shapley_sweep.py` so `python run_shapley_sweep.py 0` and `... 1`
can run independently). Verified live: launched both as background
processes simultaneously, no rate-limit conflicts, no errors -- DeepInfra
evidently doesn't share a single account-wide connection budget across
concurrent processes the way the theoretical risk suggested it might.

**Corrected the estimate with real per-model numbers** (the original
12.5-day figure conflated both models together and was badly skewed by
one 2h12m outlier run that alone accounted for ~45% of the pilot's total
time). Split by model or the outlier ONLY replaced with a generous
10-minute placeholder (the real risk noted, not eliminated -- at 160
runs instead of 30, more outliers are plausible): Qwen3-32B's full
sweep at `max_connections=20` ~1.6 days, gemma-3-27b-it's ~2.9 days.
Sequential total ~4.5 days (already a big improvement over 12.5 just
from the concurrency fix); run as two parallel processes, total is
bounded by the slower model alone: **~2.9 days**.

**Launched both processes**: `python run_shapley_sweep.py 0` (Qwen3-32B)
and `python run_shapley_sweep.py 1` (gemma-3-27b-it), each writing its
own `results/shapley_sweep_math_{model_slug}.json`, no shared state, no
collision risk. Deliberately did NOT cut seeds or `n` to hit a faster
number -- that would have undone the power analysis just completed
above; both real levers used here (concurrency, process-level
parallelism) cost nothing in statistical validity, unlike a scope cut
would have.

## Cutting the sweep further: ~2.9 days was still deemed too long, targeted budget cut on `multi_critic` configs

User still wanted the ~2.9-day estimate down, without giving up real
sample size. Laid out the actual constraint before proposing a cut:
total runtime scales with `n x seeds` (total request count), and that
product can't drop much below ~2000 without pushing `planning` (needed
N=1491/1793, the tightest currently-powered effect) below its required
threshold -- so a uniform cut has limited room (~20% max) before it
starts costing something already validated as real.

**Real asymmetry found instead: the 8 of 16 configs with
`multi_critic=True` are both the slowest in the sweep (they call a
SECOND model) and the one place the power analysis already showed the
2500-observation budget was badly mismatched to need** -- Qwen's
multi_critic main effect requires N=7416 (unreachable at ANY sane
budget, cut or not) while gemma's needs only N=587. Neither number
benefits from the full 2500 the uniform design was giving them: Qwen's
was always going to fail regardless, gemma's was already 4x
over-provisioned.

**Cut:** `MULTI_CRITIC_LIMIT = 250` (half of `LIMIT=500`), applied only
to configs with `multi_critic=True`, uniformly across both models (not
just Qwen, which is the one that actually needed it -- keeps every
compared config on equal footing, same principle as every other
generation-config decision in this project). n=250 x 5 seeds = 1250
still clears gemma's N=587 requirement with 2x margin, costs nothing on
Qwen's already-unreachable number, and stays a real, non-trivial sample
size for the interaction terms that touch `multi_critic` (not sized
directly by `power_analysis.py`, which only covers main effects, but
1250 observations is still meaningful, not negligible).

**Implementation:** `run_shapley_sweep.py` now computes `run_limit`
per-config (500 or 250) rather than a single global `LIMIT`, threads it
into `inspect_eval(limit=run_limit, ...)`, and records it per-entry in
the summary JSON (`{"seed":..., "accuracy":..., "log":..., "limit":...}`)
since it's no longer implied by one global value. Resume-loading falls
back to `LIMIT` for any pre-existing entries saved before this field
existed (correct, not a guess -- those entries genuinely were run at the
old uniform 500). One real such entry existed already: gemma's
`multi_critic` seed=1 had already completed at n=500 before this cut
was made (minimal sunk cost otherwise -- caught and killed before either
process had gotten further); left as-is rather than discarded, since a
higher-n data point is strictly fine to keep, not wrong.

Total sample-observations per model: 30,000 vs. 40,000 at the original
uniform budget (75%) -- but because the cut targets specifically the
slowest configs, the real time savings should be larger than the
sample-count ratio suggests. Revised estimate: **Qwen ~0.9-1.2 days,
gemma ~1.6-2.2 days, parallel total ~1.6-2.2 days** (down from ~2.9).
Relaunched both processes with the new config.

---

## Looking for a faster/cleaner model, and adding a third leg (Qwen2.5-72B-Instruct)

User noticed gemma was visibly outpacing Qwen3-32B and asked whether
Qwen's slow pace was fixable, then whether it was worth adding a third
model for a genuine multi-model comparison. This became a real, thorough
model search -- logging the full trail since several dead ends and one
self-correction are worth keeping for anyone revisiting model choice
later.

**Attempt 1: force Qwen3-32B to be terser via prompting.** Tested a
few-shot exemplar and a strong imperative ("your ENTIRE response must be
exactly \boxed{answer}") against the baseline MATH_SYSTEM prompt, n=15.
Both cut output tokens dramatically (633 -> 40 or 13) but also dropped
accuracy by ~20 points (0.467 -> 0.267), consistently across both
variants. Conclusion: Qwen's baseline verbosity is functioning as
informal reasoning that genuinely helps it solve harder problems, even
with explicit CoT and hidden thinking both already disabled -- forcing
terseness doesn't just save tokens, it handicaps the model's real
"bare" capability, which would corrupt the exact bare-condition
comparison this whole study depends on. Rejected; did not apply.

**Attempt 2: search DeepInfra's catalog for a replacement model.**
Queried the live `/v1/openai/models` endpoint's tag metadata for a full
picture first: 185 total models on the endpoint (includes embeddings,
image/video-gen, TTS/STT), 101 tagged `chat`, 51 text-only, and
**64 of 185 -- a majority of chat models -- tagged `reasoning`**. This
last number matters: it's not that Qwen3-32B was unlucky, hybrid
thinking-mode architecture is now the norm among available chat models,
not the exception, so any replacement search needs to specifically
filter for non-reasoning models rather than assume most models are
"normal."

Tested 8 candidates total, all via the same protocol (live headroom +
verbosity check on MATH, n=20, transcript read before trusting the
aggregate):

| model | size | verdict |
|---|---|---|
| Mistral-Small-3.2-24B-Instruct | 24B | accuracy 0.100 -- too weak (floor) |
| gemma-4-31B-it | 31B | accuracy 0.950 -- too easy (ceiling), AND verbose (657 tok) despite being gemma family |
| Nemotron-3-Nano-30B-A3B | 30B | worse verbosity than Qwen3-32B (1810 tok), 20% no-answer rate |
| microsoft/phi-4 | 14B | clean, fast, good headroom -- but **no tool-calling support at all** (hard 405 error), blocks 8/16 configs. Not a fixable bug. |
| google/gemma-3-12b-it | 12B | accuracy 0.200 -- too weak |
| mistralai/Mistral-Nemo-Instruct-2407 | 12B | accuracy 0.100 -- too weak |
| meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo | 8B | accuracy 0.050 -- too weak |
| meta-llama/Llama-3.3-70B-Instruct-Turbo | 70B | clean, tool-calling OK, good headroom -- but see below |

**Real finding, not just noise: 4 of 4 models tested in the 8-24B range
landed at 5-20% accuracy on MATH intermediate_algebra** -- consistent
enough across genuinely different model families (Mistral, gemma, Llama)
to conclude there's a real capability threshold for this specific suite
somewhere between ~24B and ~27B, not a matter of picking the right small
model. Every model that's cleared real headroom all session has been
27B+. This is itself a fact worth keeping for future suite/model
selection, independent of what it meant for the immediate search.

**Llama-3.3-70B-Instruct-Turbo: passed every check except speed, and
that took a self-correction to get right.** First comparison (different
n, different earlier test run) suggested it ran about as fast as
Qwen3-32B (56.7s vs 59.4s) -- user asked to verify this properly rather
than take it on faith, good call: a controlled, matched re-test (same
n=20, same max_connections=15, back-to-back) showed Qwen3-32B at 44.9s
vs Llama-3.3-70B at **102.5s -- genuinely ~2.3x slower**, not
comparable. The original comparison was an apples-to-oranges mistake
(different n, different run, real run-to-run variance from
temperature=0.7 sampling even at a fixed seed on this provider).
Corrected on the spot rather than left standing. Since sweep time is
bounded by the slowest leg and Qwen3-32B is already that bottleneck,
adding a model that's slower still would extend the overall sweep, not
just add a data point -- rejected for this reason specifically, not a
capability problem.

Also worth recording since it corrected a real overgeneralization made
mid-search: comparing Qwen2.5-72B (72B, 8 avg output tokens, 5.0s) to
Llama-3.3-70B (70B, 125 tokens, 56.7s in the original imprecise test)
had suggested "bigger models are slower per-token, canceling out
verbosity gains" -- **wrong**. Computing implied per-token rates from
the properly controlled numbers (Qwen2.5-72B: 12.2s/140 tokens =~
0.087s/token; Llama-3.3-70B: 102.5s/3800 tokens =~ 0.027s/token) shows
Llama's raw per-token rate isn't slower at all -- the wall-clock gap is
almost entirely explained by **output token count**, not parameter
count. Verbosity, measured directly, predicts speed far better than
model size does. Worth remembering next time a model's speed needs
estimating from its size alone.

**Qwen/Qwen2.5-72B-Instruct: passed every check.** Tool-calling confirmed
working (no 405), controlled n=20 timing at 12.2s (faster than BOTH
models already running -- won't become the bottleneck), transcripts
read directly (clean `\boxed{...}` only, no hidden reasoning, matching
the same non-reasoning generation as the project's very first validated
model, Qwen2.5-7B), headroom confirmed twice (0.300, 0.350, both n=20).
72B is a genuine third scale tier against the 27B/32B pair already
running -- 14B (phi-4) wasn't reachable due to tool-calling, so this
ended up as a small/medium -> large comparison rather than the
originally-hoped 3-tier small/medium/large, but still meaningfully
different from the current pair.

**Added as `MODEL_PAIR`'s third entry**, critic model gemma-3-27b-it
(same reasoning as every other critic-assignment decision this session:
fast, cheap, already validated, doesn't require touching the two sweeps
already in progress). Ran its pilot via `math_pilot.py` (correctly
skipped the two already-complete models, only ran the 15 new
Qwen2.5-72B runs, zero risk to the other two). Power analysis: 2 of 4
main effects well-powered at the planned budget (critique +7.5pts,
N=275 needed; multi_critic +15.8pts, N=100 needed -- both comfortably
covered by 2500 available), 2 underpowered (tool_use -3.3pts needs
N=3826; planning -0.8pts, essentially negligible, needs N=12,638) --
same acceptable pattern as the other two models, not a blocker.

**Launched as a third parallel process** (`python run_shapley_sweep.py
2`), alongside the two already running. All three now active
simultaneously, independent summary files
(`results/shapley_sweep_math_openai-api-deepinfra-qwen-qwen2.5-72b-instruct.json`),
no shared state, no collision risk.

## DeepInfra account ran out of balance mid-sweep: all three processes silently corrupted most of their remaining data, fixed at the resume-logic level

**First symptom, caught via a routine "check the runs" pass, not a crash.**
gemma's log showed 10 `Traceback` matches. Most were harmless (the
MODEL's own generated tool code hitting a `NameError`, correctly
reported back to it as a tool result -- normal `tool_use` behavior, not
a bug). One was real: a `tool_use` seed hit
`BadRequestError 400 -- max_tokens=4096 exceeds the model's 131,072
context window with 127,156 input tokens already used`, likely from the
model looping on the same broken code repeatedly without correcting
itself, ballooning the conversation until it blew the context window.
Patched the same way as every prior nan (rerun the one (config, seed)
pair, overwrite in place) -- except the very next retry hit a
**different, more serious error**: `APIStatusError 402 -- "You need
positive balance to do inference. Please add balance manually or setup
top-up"`.

**Checked whether this was isolated or account-wide before doing
anything else** -- grepped all three sweep logs, found `402` in all
three, all at the exact same wall-clock minute as the current time.
Confirmed live and ongoing, not a resolved-in-the-past blip: a smoke
test call failed the same way. This needed the user's own action (top
up DeepInfra balance) -- flagged clearly and stopped rather than attempt
a workaround.

**The real damage only became visible once the user added funds and the
sweeps were checked again: all three processes had kept running through
the ENTIRE outage and reached their final config**, each printing "Done."
-- but every request during the outage window failed and got recorded
as `accuracy: nan`, and the *original* resume logic (`if summary_path
exists, treat every recorded (config, seed) as done`) has no concept of
a failed-but-recorded entry -- a `nan` counts as "done" forever, exactly
like the earlier one-off nan patches, just at 100x the scale this time.
Counted the actual damage: **158 of 240 total seed-runs (66%) were nan**
-- Qwen3-32B 49/80, gemma 58/80, Qwen2.5-72B 51/80.

**Fix: made the resume logic itself nan-aware, instead of hand-patching
158 entries one at a time.** `run_for_model()`'s prior-results loader
now drops any `nan`-accuracy entry from both the `results` dict and the
`done` set when loading a summary file, rather than keeping it. A plain
relaunch of all three processes therefore naturally retries exactly the
158 failed (config, seed) pairs and leaves the 82 genuinely-successful
ones alone -- no risk of ending up with 6 seeds recorded for a config
that should have 5, since the stale nan entry is fully removed before
the loop re-appends a fresh one. This is a strictly better fix than
another one-off patch script: it also covers any future interruption of
the same shape (a whole-account outage, not just a single bad sample)
without needing a bespoke script each time.

**Relaunched all three processes with the fix in place.** Confirmed
recovery directly rather than assuming: within the first hour,
Qwen3-32B's nan count went from 49 to 0, gemma and Qwen2.5-72B were
actively clearing their backlogs (58 and 51 respectively), and zero new
`402` errors appeared in any log post-relaunch -- the balance top-up
fully resolved the underlying cause, and the resume-logic fix is
correctly directing all recovery effort at exactly the entries that
need it.

## Balance ran out a SECOND time; real per-model cost data revealed gemma is carrying ~2-3x the invocation load of the other two models; DeepInfra Flex tier applied selectively

**Second outage, caught while investigating an unrelated question** ("why
does gemma seem slow"). Live smoke test confirmed a fresh `402` --
balance had run out again, 34-35 occurrences already logged across all
three sweeps by the time it was caught. User topped up again ($10) and
asked for a cost estimate to finish, this time providing real DeepInfra
billing figures directly rather than reconstructing from token counts:
Qwen3-32B $9.26 spent (55.0% of total sample-work done), gemma $25.43
(30.8% done), Qwen2.5-72B $15.23 (58.3% done). Extrapolating from real
spend-per-percent-complete: **~$75.60 estimated to finish all three,
against a $10 balance -- ~$65.60 in additional funds needed.**

**Real finding, not obvious from pricing alone: gemma is the most
expensive of the three despite having the CHEAPEST per-token price**
($0.08/$0.16 vs Qwen3-32B's $0.08/$0.28 and Qwen2.5-72B's $0.36/$0.40).
Root cause is invocation volume, not price: gemma-3-27b-it is the critic
model for BOTH other sweeps (`MODEL_PAIR`'s critic column is gemma,
Qwen3-32B, gemma), so it does primary work for its own 80-run sweep AND
critic work for two other sweeps' `multi_critic` configs -- roughly 2-3x
the total invocation count of Qwen2.5-72B (which is critic for no one)
or Qwen3-32B (critic for gemma only). This was an unintended consequence
of picking gemma as Qwen2.5-72B's critic for convenience (fast, cheap,
already validated) when it was added as the third model, without
weighing the cumulative effect of gemma serving double critic duty.

**Looked for cost-cutting levers that don't change the experiment's
design before accepting the $65.60 figure.** Checked whether any of the
3 models support DeepInfra's `prompt_cache` tag (none do -- ruled out).
Found DeepInfra's "Flex" service tier instead: 0.8x standard pricing,
documented tradeoff of "slower responses and occasional unavailability,"
explicitly positioned for "non-production and asynchronous work" -- a
good match, since this sweep is already checkpointed/retry-resilient.
No native `GenerateConfig` field for it; accessible via
`extra_body={"service_tier": "flex"}`, same mechanism already used for
`enable_thinking`.

**Tested per-model before applying anywhere, not assumed uniform --
good thing, since the result was NOT uniform:** gemma-3-27b-it and
Qwen2.5-72B-Instruct both returned in ~1s on Flex tier. Qwen3-32B hit
the documented "occasional unavailability" case badly: a single request
took **1804 seconds (30 minutes)** before timing out. First test attempt
(no explicit timeout) had to be killed after 21+ minutes with zero
output before retrying with a bounded 30s timeout to get a clean
answer -- confirms this needed live verification, not just trusting the
docs' description of the tradeoff as universally mild.

**Applied selectively, not globally.** Added `FLEX_TIER_MODELS` (gemma
and Qwen2.5-72B only) and `extra_body_for(model_name)` to
`run_shapley_sweep.py`, applied both where a model is the PRIMARY model
being evaluated and where it's serving as CRITIC for another model's
`multi_critic` runs (matters specifically because gemma's critic role is
exactly where its invocation-volume problem lives). Verified the
resulting per-model extra_body dict directly before relaunching: Qwen3-32B
never gets `service_tier: flex` in either role, gemma and Qwen2.5-72B
always do. Estimated savings: gemma $57.13 -> $45.70 remaining,
Qwen2.5-72B $10.89 -> $8.71 remaining, Qwen3-32B unchanged at $7.58 --
**new total remaining estimate ~$62 (down from $75.60)**, zero change to
models, prompts, or measured behavior.

**Also considered and explicitly declined (for now) a tool_use round-cap
fix.** The repeated context-length failures on gemma's `tool_use` configs
stem from the model looping on the same broken generated code many times
without self-correcting, until the conversation balloons past the
context window -- confirmed this is the same root cause behind the
30-minute-plus wasted-time incidents. A cap on total tool-call rounds per
sample would prevent the worst-case ballooning, but unlike the earlier
`timeout=30` guard (which only bounds a single call's runaway execution,
never changes normal behavior), a round cap would be a genuine, if
narrow, change to what `tool_use` measures -- flagged to the user
explicitly rather than treated as another free infrastructure fix;
not implemented pending their decision.

**Relaunched all three processes with both the nan-retry fix and the
selective Flex tier fix together.** All three alive immediately after
restart.
