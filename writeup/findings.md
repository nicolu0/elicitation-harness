# Findings (in progress, 2 of 3 models)

**Status: partial.** This covers `Qwen2.5-72B-Instruct` and `Qwen3-32B`,
both complete (16 configs × 5 seeds × [500 or 250] samples, MATH
intermediate_algebra, DeepInfra). `gemma-3-27b-it`'s identical sweep is
still running. This is a two-model headstart. The project is scoped for
a three-model cross-model transfer result (Methodology's "Cross model
transfer" section, README's "questions we test" #3). Re-run
`shapley_attribution.py` on all three once gemma finishes, and replace
this file rather than append to it.

**Significance testing is fixed and now uses the rigorous per-sample
method.** `shapley_significance_test.py` originally crashed on this
data (`AssertionError: critique: sample-level point estimate 0.0661 !=
coalition-mean estimate 0.0689`). The cause: it assumed every coalition
shares one fixed `n` (true for the old GPQA sweep, false here. The 8
`multi_critic=True` configs run at `n=250` while the other 8 run at
`n=500`). The fix computes every coalition's mean accuracy restricted
to the sample IDs shared across all 16 coalitions an effect touches.
Every main effect's linear combination spans all 16 coalitions, so
that shared set is capped at 250 for every single effect, including
ones that don't name `multi_critic`. This follows the same principle
`mcnemar_test.py` already uses throughout this project: paired
comparisons need the identical question set, which is a stricter
requirement than matching the question count. This restricted method
is now the primary result below. The seed-bootstrap numbers from the
first pass are kept alongside for reference, but they're the weaker of
the two methods (per the GPQA-era precedent, where the bootstrap
overturned 2 of 3 "significant" findings once the sample-level test
ran).

**Multiple comparisons.** A Benjamini-Hochberg FDR correction at
α=0.05 was applied across all 20 tests (4 main effects + 6 interactions
× 2 models), using the Wilcoxon p-value from the sample-level test.
Every finding that looked significant before the correction remains
significant after it: all 8 main effects (4 per model) and 6 of 12
interactions (3 per model). The p-values that cleared 0.05 uncorrected
were almost all far below it (most <0.0001), so the correction confirms
these results rather than narrowing them. Full ranked table in the
significance section below.

**Validity-hazard check.** This is a real finding, and it isn't yet in
`phases.md`'s ledger. Twelve random bare-condition transcripts per
model were read (the full Phase 4 standard asks for at least 30; this
is a partial check, flagged as such).

| model | refusals | visible CoT leaking into bare (system prompt says "do not show your work") |
|---|---|---|
| Qwen2.5-72B-Instruct | 0/12 | 0/12. Every sampled answer was a bare `\boxed{...}`, 9-20 characters |
| Qwen3-32B | 0/12 | **8/12**. Full step-by-step derivations, up to 8,329 characters, despite `enable_thinking: false` already applied |

process_log.md already documented this reasoning-mode issue and
believed it fixed (`enable_thinking: false` cut one hard question's
output from 1977 to 5 tokens in an earlier smoke test). The fix
evidently only suppresses the *hidden* thinking channel. The model
still frequently ignores "do not show your work" and reasons directly
in its visible reply. This matches the Phase 4 hazard by name:
chain-of-thought leaking into the no-CoT condition. **Practical
consequence: Qwen3-32B's "bare" baseline (0.564) is likely inflated
above its true bare capability by free CoT-like reasoning that slipped
past the system prompt.** See the steelman section below for what this
means for the headline finding.

---

## Coalition means (accuracy, mean of 5 seeds)

**Qwen2.5-72B-Instruct**: bare 0.263 → full-stack (all 4 on) 0.636.
Every intermediate coalition sits between those two, increasing
monotonically with more components on (full ranking in
`shapley_attribution.py`'s output). Total lift +0.373.

**Qwen3-32B**: bare 0.564 → full-stack 0.573. Total lift +0.009, inside
noise. The single best coalition is `critique+multi_critic` alone
(0.625), without `tool_use` or `planning`. Adding the other two
components on top makes it worse.

## Shapley main effects, sample-level paired test (primary result)

Point estimate restricted to the 1250 shared (sample, seed) pairs
common to all 16 coalitions (see the significance-testing note above
for why). `p` is the Wilcoxon signed-rank p-value. FDR marks whether
the finding survives Benjamini-Hochberg correction across all 20
tests, α=0.05.

| component | Qwen2.5-72B-Instruct | p | FDR | Qwen3-32B | p | FDR |
|---|---|---|---|---|---|---|
| `multi_critic` | **+0.161** | <0.0001 | ✅ | **+0.073** | <0.0001 | ✅ |
| `tool_use` | **+0.107** | <0.0001 | ✅ | **−0.083** | <0.0001 | ✅ |
| `critique` | **+0.066** | <0.0001 | ✅ | **−0.016** | 0.0052 | ✅ |
| `planning` | **+0.028** | <0.0001 | ✅ | **+0.021** | 0.0021 | ✅ |

All 8 main effects, on both models, survive FDR correction. This is a
stronger result than the earlier seed-bootstrap pass: none of it rests
on the weaker of the project's two significance methods anymore.

## Pairwise interactions, sample-level paired test (primary result)

| pair | Qwen2.5-72B-Instruct | p | FDR | Qwen3-32B | p | FDR |
|---|---|---|---|---|---|---|
| `tool_use` × `multi_critic` | **−0.121** | <0.0001 | ✅ | **+0.067** | <0.0001 | ✅ |
| `critique` × `multi_critic` | **+0.023** | 0.0095 | ✅ | **+0.150** | <0.0001 | ✅ |
| `tool_use` × `planning` | **+0.044** | <0.0001 | ✅ | +0.022 | 0.0706 | no |
| `critique` × `tool_use` | −0.002 | 0.9569 | no | **+0.050** | <0.0001 | ✅ |
| `critique` × `planning` | −0.003 | 0.8987 | no | +0.004 | 0.7352 | no |
| `planning` × `multi_critic` | −0.005 | 0.6689 | no | −0.006 | 0.5992 | no |

3 of 6 interactions per model survive. `planning` × `multi_critic` and
`critique` × `planning` are clean nulls on both models: genuinely no
interaction, not underpowered noise. `tool_use` × `multi_critic` is
the standout. It's strongly negative on Qwen2.5-72B and positive on
Qwen3-32B, significant both times, opposite signs. That's a confirmed
interaction-level rank flip between the two models, beyond the
main-effect flip already noted above.

**Seed-bootstrap CIs (secondary, kept for reference).** Point estimates
agree with the sample-level test to within roughly 0.005-0.015 in every
case. That's expected: the bootstrap uses each coalition's own full
`n`, while the sample-level test restricts to the shared 250-sample
intersection (see the significance-testing note above). Full
seed-bootstrap numbers are in `shapley_attribution.py`'s own output if
needed; they aren't reproduced here since the sample-level test above
supersedes them.

---

## The four things we test, status on 2 of 3 models

**1. Credit is concentrated, and the parts aren't independent.** True
for Qwen2.5-72B: `multi_critic` + `tool_use` alone account for +0.277
of the +0.373 total lift (74%), and there's a real, signed interaction
structure (`tool_use`×`multi_critic` strongly negative,
`critique`×`multi_critic` positive). Qwen3-32B tells a different story:
total lift is statistically indistinguishable from zero, so "credit
concentration" isn't a meaningful question when there's barely any
credit to concentrate. The two models disagree on whether this claim
even applies, and that disagreement is itself the finding.

**2. Components and budget substitute for each other.** Not tested.
Budget isn't an axis in this study yet (README's own "Limitations"
section already states this).

**3. The ranking may not transfer, the one we care most about.** This
is the headline, and it's a clean divergence. `tool_use` has the
**opposite sign** between the two models (+0.107 vs −0.083, both
significant, both survive FDR correction), and `tool_use`×`multi_critic`
also flips sign (−0.121 vs +0.067, also both significant).
Statistically this is now about as solid as it gets on this dataset:
a rigorous per-sample test, FDR-corrected, both directions clearly
nonzero. Statistical significance is a different question from
validity, though. See the steelman section for why the sign itself,
not just its existence, is still in question for Qwen3-32B
specifically.

**4. Head-to-head comparisons can flip.** Can't fully test yet (needs
gemma for a real 3-way comparison), but there's already a hint within
just these two. Qwen3-32B's bare accuracy (0.564) beats Qwen2.5-72B's
bare accuracy (0.263) by a wide margin, while Qwen2.5-72B's full-stack
accuracy (0.636) edges past Qwen3-32B's full-stack accuracy (0.573).
That's a real rank flip depending on how much harness you apply. See
the steelman section for why this specific flip is the one most likely
to be a validity artifact rather than a real finding.

## Steelman the null

**For claim #3 (tool_use's sign flip is a real capability-transfer
finding):** the steelman-the-null case is strong here. Qwen3-32B's
bare condition is contaminated by leaked CoT in a majority (8/12) of
sampled transcripts. A model that's already reasoning its way to the
answer in the "bare" condition has much less room left for
`tool_use`/`critique`/`planning` to add anything. Every additional turn
(a tool call, a critique pass) is also another opportunity for the
already-present reasoning-leakage tendency to derail into the kind of
degenerate loop documented in the process log's `tool_use`+`planning`
entry (500+ identical repeated tool calls). The honest read: this
result is at least partly, and maybe mostly, explained by Qwen3-32B's
bare baseline being artificially inflated by a system-prompt-compliance
failure. A cleaner test needs a bare condition that's actually enforced
(for example, reject/retry any bare-condition completion that isn't a
short boxed answer) before the sign flip can be trusted as a genuine
cross-model transfer finding, rather than one model simply breaking a
formatting instruction more than the other.

**For claim #4 (the bare-vs-full-stack rank flip):** the same caveat
applies, more strongly. Qwen3-32B's bare-condition "win" over
Qwen2.5-72B is substantially built on free CoT the system prompt was
supposed to suppress, so it isn't a clean apples-to-apples
"capability without scaffolding" comparison. This specific flip is the
*least* trustworthy finding in this document until the bare condition
is fixed.

**For the multi_critic finding (positive on both models, the largest
single effect on both, surviving every test applied so far):** this
one holds up best under the same scrutiny. `multi_critic` is a
genuinely separate generation channel. It calls the critic model, not
the primary model's own extended reasoning, so bare-condition CoT
leakage has no obvious mechanism to explain why it helps both models
by a wide, clearly-nonzero, FDR-corrected margin. This is the
steelman-resistant claim in this dataset, the one most likely to still
be real once the CoT-leak issue is fixed.

## What would change this writeup

Two of the original four action items are done, on the statistical
rigor side. The two that touch validity and scope are still open.

1. ~~Fix `shapley_significance_test.py` for non-uniform per-config `n`~~.
   **Done.** See the significance-testing note and tables above.
2. ~~Apply a real multiple-comparisons correction~~. **Done.** BH-FDR
   across all 20 tests, results in the tables above.
3. **Still open.** Either enforce the bare-condition format (reject/retry
   non-compliant completions) or explicitly quantify the CoT-leak rate
   across the full dataset (not just 12 samples), and report bare
   accuracy both as-is and with leaked-CoT samples excluded, so the
   `tool_use` sign-flip claim can be checked against a clean baseline.
   This is the single biggest remaining threat to trusting claim #3 as
   a genuine cross-model transfer finding rather than a validity
   artifact.
4. **Still open.** Add gemma once its sweep finishes. This is 2 of 3
   models, and the project's own framing (README: "similar scale,
   different families") means the real transfer claim needs the third
   leg. `shapley_significance_test.py`'s fix above applies unchanged to
   gemma's data (same non-uniform-`n` issue, same fix).
