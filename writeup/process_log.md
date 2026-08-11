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
      cut), 5 seeds, Qwen2.5-7B + Llama-3.1-8B (Together), GPQA only for
      now. Estimated cost ~$45.58, real pricing pulled live.
- [ ] `best_of_n` toggle — NOT BUILT, deliberately deferred (needs the
      budget axis to show its actual value, per the scope decision above)
- [ ] `multi_critic` toggle — NOT BUILT yet, next component to implement
- [ ] Pilot the 4 real components (`tool_use`, `planning`, `multi_critic`
      empirically untested; `best_of_n` excluded) at small scale
      (n=20-40, 1 seed) to replace the estimated per-component token
      costs with measured ones before committing to the full ~$46 sweep
- [ ] Full 4-component (2⁴=16 config) × 5-seed × 2-model Shapley sweep on
      GPQA — not started; blocked on the components above being built
      and the pilot validating the cost estimate
