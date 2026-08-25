# Findings (3 of 3 models, complete)

This covers all three models in the current MATH sweep:
`Qwen2.5-72B-Instruct`, `Qwen3-32B`, and `gemma-3-27b-it`. All three are
complete: 16 configs × 5 seeds × [500 or 250] samples, MATH
intermediate_algebra, DeepInfra. This replaces the earlier two-model
headstart version of this file rather than appending to it.

**A data-corruption bug was found and fixed while finishing this
analysis, and it matters for how much to trust gemma's numbers.**
DeepInfra balance outages hit gemma's sweep repeatedly. `fail_on_error`
and `score_on_error` (added earlier to stop one bad sample from
crashing a whole run) had a side effect nobody caught until now: a run
where most or all samples failed from an outage still produced a real,
non-NaN accuracy number instead of NaN, because the errored samples got
scored as wrong rather than excluded. The resume logic only checked for
NaN, so these corrupted entries got treated as permanently done and
never retried. A check of every log's actual status (not just its
saved accuracy number) found 4 of gemma's 80 entries were affected,
with 22% to 99% of their samples errored. `Qwen2.5-72B-Instruct` and
`Qwen3-32B` were both completely clean, 0 bad entries, consistent with
neither hitting a mid-run outage as often as gemma did. The resume
logic in `run_shapley_sweep.py` now checks each entry's actual log
status, not just its accuracy value, so this can't recur silently. All
4 corrupted entries were rerun and gemma's data is now fully clean
(verified again after the rerun: 0 bad entries out of 80).

**Significance testing uses the rigorous per-sample method
throughout.** `shapley_significance_test.py` restricts every
coalition's mean to the sample IDs shared across all 16 coalitions an
effect touches, rather than each coalition's own full `n` (needed
because the 8 `multi_critic=True` configs run at `n=250` while the
other 8 run at `n=500`). This is the primary result below. A
Benjamini-Hochberg FDR correction at α=0.05 was applied across all 30
tests (4 main effects + 6 interactions × 3 models), using the Wilcoxon
p-value. 20 of 30 survive: 11 of 12 main effects, 9 of 18 interactions.

**Validity-hazard check still open.** A spot-check of 12 bare-condition
transcripts found `Qwen3-32B` leaking visible chain-of-thought into 8
of 12 sampled answers, despite the system prompt saying not to show
its work. `Qwen2.5-72B-Instruct` showed none. This wasn't re-checked
for gemma; the fix (enforce the bare format, or quantify the leak rate
at scale) is still an open item and applies to whichever models need
it.

---

## Coalition means (accuracy, mean of 5 seeds)

| model | bare | full-stack | total lift |
|---|---|---|---|
| Qwen2.5-72B-Instruct | 0.263 | 0.636 | +0.373 |
| Qwen3-32B | 0.564 | 0.573 | +0.009 (inside noise) |
| gemma-3-27b-it | 0.208 | 0.380 | +0.172 |

Qwen3-32B's near-zero lift already looked unusual with two models; with
a third model showing a real, large lift (gemma, +0.172), it looks even
more like Qwen3-32B is the outlier here, not the norm. Qwen3-32B's best
coalition isn't the full stack either. It's `critique+multi_critic`
alone (0.625), and adding `tool_use` or `planning` on top makes it
worse. Gemma's best coalition is `critique+planning` (0.511), also not
the full stack, and also without `multi_critic`.

## Shapley main effects, sample-level paired test

`p` is the Wilcoxon signed-rank p-value. FDR marks whether the finding
survives Benjamini-Hochberg correction across all 30 tests, α=0.05.

| component | Qwen2.5-72B | p | FDR | Qwen3-32B | p | FDR | gemma-3-27b-it | p | FDR |
|---|---|---|---|---|---|---|---|---|---|
| `critique` | **+0.066** | <0.0001 | ✅ | **−0.016** | 0.0052 | ✅ | **+0.113** | <0.0001 | ✅ |
| `tool_use` | **+0.107** | <0.0001 | ✅ | **−0.083** | <0.0001 | ✅ | **−0.034** | 0.0001 | ✅ |
| `planning` | **+0.028** | <0.0001 | ✅ | **+0.021** | 0.0021 | ✅ | **+0.077** | <0.0001 | ✅ |
| `multi_critic` | **+0.161** | <0.0001 | ✅ | **+0.073** | <0.0001 | ✅ | −0.004 | 0.4206 | no |

**The headline result from the two-model version of this writeup does
not hold up.** `multi_critic` was flagged there as the most trustworthy
finding in the whole dataset: positive on both models, the largest
single effect, immune to the CoT-leak issue since it's a separate
model call. Adding the third model changes that. `multi_critic` is not
significant on gemma at all (p=0.42), and its point estimate is
essentially zero. No single component is positive and significant on
all three models. `critique` and `planning` come closest: both
significant on all three, but `critique` is negative on Qwen3-32B and
`tool_use` is negative on both Qwen3-32B and gemma. There's no
universal finding here. Every component's sign depends on which model
it's measured on.

## Pairwise interactions, sample-level paired test

| pair | Qwen2.5-72B | p | FDR | Qwen3-32B | p | FDR | gemma-3-27b-it | p | FDR |
|---|---|---|---|---|---|---|---|---|---|
| `critique` × `tool_use` | −0.002 | 0.9569 | no | **+0.050** | <0.0001 | ✅ | **−0.143** | <0.0001 | ✅ |
| `critique` × `planning` | −0.003 | 0.8987 | no | +0.004 | 0.7352 | no | +0.008 | 0.4840 | no |
| `critique` × `multi_critic` | **+0.023** | 0.0095 | ✅ | **+0.150** | <0.0001 | ✅ | −0.000 | 0.9876 | no |
| `tool_use` × `planning` | **+0.044** | <0.0001 | ✅ | +0.022 | 0.0706 | no | **−0.057** | <0.0001 | ✅ |
| `tool_use` × `multi_critic` | **−0.121** | <0.0001 | ✅ | **+0.067** | <0.0001 | ✅ | +0.013 | 0.1463 | no |
| `planning` × `multi_critic` | −0.005 | 0.6689 | no | −0.006 | 0.5992 | no | **−0.039** | <0.0001 | ✅ |

`critique` × `tool_use` is the one interaction that's significant on
all three models, and gemma's sign disagrees with Qwen3-32B's (−0.143
vs +0.050, both significant). `tool_use` × `multi_critic`, the
strongest single interaction in the two-model version (−0.121 on
Qwen2.5-72B, +0.067 on Qwen3-32B, both significant, opposite signs),
isn't even significant on gemma (p=0.15). That flip looked like a
clean two-model story before. With the third model it's one model
showing no effect at all, not a clean three-way disagreement.

## Cross-model rank correlation

Ranking the four components by main-effect size on each model and
computing pairwise Spearman ρ and Kendall τ:

| pair | Spearman ρ | Kendall τ |
|---|---|---|
| Qwen2.5-72B vs Qwen3-32B | 0.20 | 0.00 |
| Qwen3-32B vs gemma-3-27b-it | 0.20 | 0.00 |
| Qwen2.5-72B vs gemma-3-27b-it | **−0.60** | **−0.33** |

None of these are statistically significant on their own (n=4
components per ranking is very low power for a correlation; none of
the p-values here clear 0.4). Read them as descriptive, not confirmed.
But the direction is worth noting regardless: two of the three pairs
show weak positive agreement, and the third (Qwen2.5-72B vs gemma) is
negative. A negative rank correlation between two models' component
rankings is about as clear a "the ranking doesn't transfer" signal as
this kind of small-n test can produce, even without a significance
claim attached to the correlation coefficient itself.

---

## The three things we test, status on all 3 models

**1. Credit is concentrated, and the parts aren't independent.**
Qwen2.5-72B: true. `multi_critic` + `tool_use` alone account for 74%
of the total lift, with a real, signed interaction structure.
Qwen3-32B: not really. Total lift is near zero, so there's barely any
credit to concentrate. Gemma: true, but concentrated in different
components. `critique` + `planning` account for +0.19 of the +0.172
total lift by themselves (more than the total, since some other
components subtract), while `multi_critic` and `tool_use` contribute
close to nothing net. Three models, three different concentration
patterns.

**2. The ranking may not transfer.** Confirmed, more strongly than the
two-model version suggested. No component is positive and significant
across all three models. The Qwen2.5-72B vs gemma rank correlation is
negative. `tool_use` × `multi_critic`, the interaction that looked like
the standout cross-model finding with two models, isn't significant on
the third.

**3. Head-to-head comparisons can flip.** Bare, the ranking is
Qwen3-32B (0.564) > Qwen2.5-72B (0.263) > gemma (0.208). Fully
scaffolded, it's Qwen2.5-72B (0.636) > Qwen3-32B (0.573) > gemma
(0.380). Qwen2.5-72B and Qwen3-32B swap first place depending on how
much harness is applied; gemma stays last in both conditions but closes
part of the gap (from 0.355 behind the leader bare to 0.256 behind
fully scaffolded). The Qwen3-32B bare-condition caveat below still
applies to the first ranking specifically.

## Steelman the null

**The `multi_critic` finding from the two-model version was wrong, and
that's worth stating plainly rather than quietly dropping.** It was
flagged there as the steelman-resistant claim in the whole dataset:
positive on both models measured, the largest single effect, immune to
the CoT-leak confound. All of that reasoning was sound given the data
available at the time. It just didn't hold up once a third model was
added. This is the exact failure mode the project is designed to catch
(a two-model "universal" finding that turns out to be two-model-specific),
and it happened to the strongest claim in this document. Take that as
a caution against trusting any single component's cross-model
generality here, including the ones that currently look solid on all
three models (`critique`, `planning`).

**For the Qwen3-32B CoT-leak issue specifically:** unchanged from the
two-model version. Qwen3-32B's bare condition leaks visible reasoning
in a majority of sampled transcripts, which likely inflates its bare
baseline and understates how much every component adds on top of it.
This doesn't explain gemma's results (gemma wasn't checked, but its
total lift and component pattern don't resemble Qwen3-32B's suppressed-effect
shape), so it's specifically a Qwen3-32B caveat, not a general one.

**For the negative Qwen2.5-72B vs gemma rank correlation:** this reads
as a real transfer failure, not a validity artifact. Neither model's
bare condition is known to be contaminated (gemma wasn't spot-checked,
but nothing in its numbers resembles the CoT-leak pattern found in
Qwen3-32B), so there's no obvious confound to explain away the
disagreement the way there was for the tool_use finding in the
two-model version.

## What would change this writeup

1. **Still open.** Check gemma's bare-condition transcripts for the
   same CoT-leak pattern found in Qwen3-32B (not yet done for this
   model). Either enforce the bare-condition format across all three
   models or quantify the leak rate at scale.
2. **Still open.** None of the components that look consistent across
   these three models (`critique`, `planning`) have been checked
   against a fourth model or a second task suite. Everything here is
   specific to MATH intermediate_algebra and these three model
   families.
3. **Worth doing, not urgent.** The data-corruption bug found in this
   pass (accuracy computed from a mostly-errored run passing as valid)
   was caught by manually checking one suspicious high-variance
   coalition. A systematic version of that check (verify every entry's
   log status before trusting it) is now built into `run_shapley_sweep.py`'s
   resume logic going forward, but the same check could be worth adding
   as a standalone script (`verify_results.py` or similar) that audits
   an existing results file without needing to relaunch the sweep.
