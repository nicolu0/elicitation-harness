"""
Paired significance test (McNemar's) between two Inspect eval log files.

Since every config in run_ablation.py runs on the IDENTICAL 200 samples
(shuffle=False), you have PAIRED data: for each task, you know whether
config A got it right/wrong AND whether config B got it right/wrong on
that exact same task. That pairing is statistical power you should use --
comparing two independent accuracy +/- stderr numbers throws it away.

McNemar's test only looks at the samples where the two configs DISAGREED
(one got it right, the other wrong) and asks whether the disagreements are
lopsided in one direction more than chance would predict.

Install (one-time):
    uv pip install scipy

Usage:
    1. Find the two log files you want to compare in ./logs/
       (run_ablation.py printed the log path after each config, e.g.
       "cot: True, critique: False" -> logs/2026-07-23T18-36-21..._elicit_....eval)
    2. python mcnemar_test.py logs/<cot_log>.eval logs/<cot_critique_log>.eval
"""

import sys

from inspect_ai.log import read_eval_log
from scipy.stats import binomtest


def sample_outcomes(log_path: str) -> dict[str, bool]:
    """Return {sample_id: correct} for every sample in a log file."""
    log = read_eval_log(log_path)
    outcomes = {}
    for sample in log.samples:
        # scorer key varies by suite: "match" (gsm8k/math), "letter_match" (gpqa)
        scorer_key = next(iter(sample.scores))
        score = sample.scores[scorer_key]
        value = score.value
        correct = value in ("C", 1, 1.0, True)
        outcomes[str(sample.id)] = correct
    return outcomes


def mcnemar(a_path: str, b_path: str, label_a: str = "A", label_b: str = "B"):
    a = sample_outcomes(a_path)
    b = sample_outcomes(b_path)

    shared_ids = set(a) & set(b)
    if len(shared_ids) < len(a) or len(shared_ids) < len(b):
        print(
            f"WARNING: only {len(shared_ids)} shared sample ids "
            f"(A has {len(a)}, B has {len(b)}). Results may not be paired "
            f"correctly -- confirm both configs ran with shuffle=False on "
            f"the same dataset slice."
        )

    # The four McNemar cells, restricted to shared samples
    both_right = both_wrong = a_only = b_only = 0
    for sid in shared_ids:
        ca, cb = a[sid], b[sid]
        if ca and cb:
            both_right += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca and not cb:
            a_only += 1   # A right, B wrong -- A "won" this sample
        else:
            b_only += 1   # B right, A wrong -- B "won" this sample

    n_disagree = a_only + b_only
    acc_a = sum(a.values()) / len(a)
    acc_b = sum(b.values()) / len(b)

    print(f"\n{label_a}: {a_path}")
    print(f"{label_b}: {b_path}\n")
    print(f"n shared samples     : {len(shared_ids)}")
    print(f"accuracy {label_a:<12}: {acc_a:.3f}")
    print(f"accuracy {label_b:<12}: {acc_b:.3f}")
    print(f"raw delta            : {acc_b - acc_a:+.3f}\n")

    print("McNemar 2x2 (restricted to disagreements):")
    print(f"  both correct        : {both_right}")
    print(f"  both incorrect      : {both_wrong}")
    print(f"  {label_a} right / {label_b} wrong : {a_only}")
    print(f"  {label_b} right / {label_a} wrong : {b_only}")
    print(f"  total disagreements : {n_disagree}\n")

    if n_disagree == 0:
        print("No disagreements between configs -- nothing to test, "
              "the two configs produced identical results on every sample.")
        return

    # Exact McNemar test = two-sided binomial test on the discordant pairs,
    # testing whether a_only is drawn from Binomial(n_disagree, 0.5)
    result = binomtest(a_only, n_disagree, 0.5, alternative="two-sided")
    p_value = result.pvalue

    print(f"McNemar exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        winner = label_b if b_only > a_only else label_a
        print(f"=> SIGNIFICANT at p<0.05: {winner} is reliably different "
              f"from {label_a if winner == label_b else label_b} on this "
              f"data (not just sampling noise).")
    else:
        print(f"=> NOT significant at p<0.05: the difference between "
              f"{label_a} and {label_b} is consistent with random noise "
              f"at this sample size. Do not report this as a real effect.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    mcnemar(sys.argv[1], sys.argv[2], label_a="config_A", label_b="config_B")
