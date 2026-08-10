"""
Paired significance test (McNemar's) between two configs, either from a
single pair of Inspect eval logs, or POOLED across multiple seeds (see
run_ablation.py's temperature/seed sweep).

Since every config in run_ablation.py runs on the IDENTICAL samples
(shuffle=False), you have PAIRED data: for each task, you know whether
config A got it right/wrong AND whether config B got it right/wrong on
that exact same task. That pairing is statistical power you should use --
comparing two independent accuracy +/- stderr numbers throws it away.

McNemar's test only looks at the samples where the two configs DISAGREED
(one got it right, the other wrong) and asks whether the disagreements are
lopsided in one direction more than chance would predict.

POOLED MODE: when the underlying dataset is capped (e.g. GPQA Diamond's
198 questions), you can't get more disagreements by raising --limit. But
if you ran multiple seeds at temperature > 0 (see run_ablation.py), each
seed gives you a fresh, independently-sampled set of (dis)agreements on
the SAME questions. Pooled mode sums the 2x2 McNemar table across every
seed pair before running the test, which is the correct way to combine
them for more power. Caveat: repeated seeds on the same fixed question set
aren't fully independent draws, so treat the resulting p-value as an
approximation, not textbook-exact -- say so if you report it.

Install (one-time):
    uv pip install scipy

Usage (single run):
    python mcnemar_test.py logs/<config_a_log>.eval logs/<config_b_log>.eval

Usage (pooled across seeds):
    python mcnemar_test.py --pooled results/ablation_summary.json bare cot
    (reads the log paths run_ablation.py already saved per config/seed)
"""

import json
import sys

from inspect_ai.log import read_eval_log
from scipy.stats import binomtest

# Every adapter in elicit_task.py registers its scorer under a different
# name (match / boxed_match / letter_match / ...). Try each known name
# rather than hardcoding one, so this script works across every suite.
KNOWN_SCORER_NAMES = ["match", "boxed_match", "letter_match", "choice"]


def sample_outcomes(log_path: str) -> dict[str, bool]:
    """Return {sample_id: correct} for every sample in a log file, using
    whichever registered scorer produced the scores."""
    log = read_eval_log(log_path)
    outcomes = {}
    for sample in log.samples:
        score = None
        for name in KNOWN_SCORER_NAMES:
            if name in sample.scores:
                score = sample.scores[name]
                break
        if score is None:
            # fall back to whatever scorer is present, if only one is
            (score,) = sample.scores.values() if len(sample.scores) == 1 else (None,)
        if score is None:
            raise KeyError(
                f"Couldn't find a known scorer in {log_path} sample "
                f"{sample.id}. Scorers present: {list(sample.scores)}. "
                f"Add the name to KNOWN_SCORER_NAMES above."
            )
        correct = score.value in ("C", 1, 1.0, True)
        outcomes[str(sample.id)] = correct
    return outcomes


def pairwise_table(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int, int]:
    """Return (both_right, both_wrong, a_only, b_only) over shared sample ids."""
    shared_ids = set(a) & set(b)
    both_right = both_wrong = a_only = b_only = 0
    for sid in shared_ids:
        ca, cb = a[sid], b[sid]
        if ca and cb:
            both_right += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca and not cb:
            a_only += 1
        else:
            b_only += 1
    return both_right, both_wrong, a_only, b_only


def report(both_right, both_wrong, a_only, b_only, label_a, label_b, n_a, n_b):
    n_disagree = a_only + b_only
    print("McNemar 2x2 (restricted to disagreements):")
    print(f"  both correct        : {both_right}")
    print(f"  both incorrect      : {both_wrong}")
    print(f"  {label_a} right / {label_b} wrong : {a_only}")
    print(f"  {label_b} right / {label_a} wrong : {b_only}")
    print(f"  total disagreements : {n_disagree}\n")

    if n_disagree == 0:
        print("No disagreements between configs -- nothing to test.")
        return

    result = binomtest(a_only, n_disagree, 0.5, alternative="two-sided")
    p_value = result.pvalue

    print(f"McNemar exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        winner = label_b if b_only > a_only else label_a
        loser = label_a if winner == label_b else label_b
        print(f"=> SIGNIFICANT at p<0.05: {winner} is reliably different "
              f"from {loser} on this data (not just sampling noise).")
    elif p_value < 0.10:
        print(f"=> Not significant at p<0.05, but clears p<0.10 -- a "
              f"trend, not yet a confirmed effect.")
    else:
        print(f"=> NOT significant: the difference between {label_a} and "
              f"{label_b} is consistent with random noise at this sample "
              f"size. Do not report this as a real effect.")


def mcnemar_single(a_path: str, b_path: str, label_a="A", label_b="B"):
    a = sample_outcomes(a_path)
    b = sample_outcomes(b_path)
    shared = set(a) & set(b)
    if len(shared) < len(a) or len(shared) < len(b):
        print(f"WARNING: only {len(shared)} shared sample ids "
              f"(A has {len(a)}, B has {len(b)}).")

    print(f"\n{label_a}: {a_path}")
    print(f"{label_b}: {b_path}\n")
    print(f"n shared samples     : {len(shared)}")
    print(f"accuracy {label_a:<12}: {sum(a.values())/len(a):.3f}")
    print(f"accuracy {label_b:<12}: {sum(b.values())/len(b):.3f}\n")

    both_right, both_wrong, a_only, b_only = pairwise_table(a, b)
    report(both_right, both_wrong, a_only, b_only, label_a, label_b, len(a), len(b))


def mcnemar_pooled(summary_path: str, label_a: str, label_b: str):
    with open(summary_path) as f:
        summary = json.load(f)

    runs_a = summary["results"].get(label_a)
    runs_b = summary["results"].get(label_b)
    if runs_a is None or runs_b is None:
        raise KeyError(
            f"Config labels must be one of {list(summary['results'])}, "
            f"got '{label_a}' and '{label_b}'."
        )
    if len(runs_a) != len(runs_b):
        print(f"WARNING: {label_a} has {len(runs_a)} seed runs, {label_b} "
              f"has {len(runs_b)}. Pairing by position; mismatches may "
              f"pair the wrong seeds together.")

    print(f"Pooling {min(len(runs_a), len(runs_b))} seed-pairs for "
          f"'{label_a}' vs '{label_b}'\n")

    totals = [0, 0, 0, 0]  # both_right, both_wrong, a_only, b_only
    accs_a, accs_b = [], []
    for ra, rb in zip(runs_a, runs_b):
        a = sample_outcomes(ra["log"])
        b = sample_outcomes(rb["log"])
        accs_a.append(ra["accuracy"])
        accs_b.append(rb["accuracy"])
        cell = pairwise_table(a, b)
        totals = [t + c for t, c in zip(totals, cell)]
        print(f"  seed {ra['seed']:>3}: {label_a}={ra['accuracy']:.3f}  "
              f"{label_b}={rb['accuracy']:.3f}  "
              f"disagreements this seed: {cell[2] + cell[3]}")

    print(f"\nmean accuracy {label_a:<12}: {sum(accs_a)/len(accs_a):.3f}")
    print(f"mean accuracy {label_b:<12}: {sum(accs_b)/len(accs_b):.3f}\n")

    report(*totals, label_a, label_b, None, None)
    print(
        "\nNote: pooled across seeds on the same fixed question set -- "
        "not fully independent draws, treat p-value as approximate."
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--pooled"]:
        if len(args) != 4:
            print("Usage: python mcnemar_test.py --pooled <summary.json> <label_a> <label_b>")
            sys.exit(1)
        _, summary_path, label_a, label_b = args
        mcnemar_pooled(summary_path, label_a, label_b)
    elif len(args) == 2:
        mcnemar_single(args[0], args[1], label_a="config_A", label_b="config_B")
    else:
        print(__doc__)
        sys.exit(1)
