"""
Pull a random sample of transcripts from an Inspect eval log into a
markdown worksheet for manual scorer spot-checking -- Phase 2's
requirement: >=30 transcripts per suite, with a counted false-positive
and false-negative rate, not just "it ran without erroring."

Deliberately does NOT try to judge correctness itself -- that would just
be trusting a second piece of code to grade the first. This only pulls a
*random* (not cherry-picked) sample and lays out everything needed for a
human to judge: the question, the model's full completion, the extracted
answer, the target, and the scorer's verdict. You read each one and fill
in the checkbox yourself.

Usage:
    python spot_check.py logs/<run>.eval --n 30 --seed 0 \\
        --out writeup/spot_check_math.md

After filling in the worksheet, count how many entries you marked
"scorer is WRONG" split into:
  - false negative: scorer said I, you say the answer was actually right
  - false positive: scorer said C, you say the answer was actually wrong
Both counts (plus the total N reviewed) are what goes in process_log.md.
"""

import argparse
import random

from inspect_ai.log import read_eval_log

KNOWN_SCORER_NAMES = ["match", "boxed_match", "letter_match", "choice"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path")
    ap.add_argument("--n", type=int, default=30, help="sample size")
    ap.add_argument("--seed", type=int, default=0, help="for a reproducible sample")
    ap.add_argument("--out", default=None, help="output .md path")
    args = ap.parse_args()

    log = read_eval_log(args.log_path)
    samples = log.samples
    rng = random.Random(args.seed)
    chosen = rng.sample(samples, min(args.n, len(samples)))

    lines = [
        f"# Scorer spot-check: `{args.log_path}`",
        "",
        f"{len(chosen)} randomly sampled transcripts (seed={args.seed}, "
        f"out of {len(samples)} total). For each: read the model's full "
        f"completion, decide for yourself whether the scorer's verdict is "
        f"right, and fill in the checkbox. Don't skip ones that 'look "
        f"obviously fine' at a glance -- that's exactly how a scorer bug "
        f"survives review.",
        "",
    ]

    for i, sample in enumerate(chosen, 1):
        score = None
        for name in KNOWN_SCORER_NAMES:
            if name in sample.scores:
                score = sample.scores[name]
                break
        verdict = score.value if score else "?"
        model_answer = score.answer if score else "?"

        lines.append(f"## {i}. sample `{sample.id}` — scorer said: **{verdict}**")
        lines.append(f"**Question:**\n```\n{sample.input}\n```")
        lines.append(f"**Model completion:**\n```\n{sample.output.completion}\n```")
        lines.append(f"**Extracted answer:** `{model_answer}`")
        lines.append(f"**Target:** `{sample.target}`")
        lines.append(
            "**Your verdict:** [ ] agree with scorer   "
            "[ ] scorer is WRONG (say why below)"
        )
        lines.append("")
        lines.append("")

    out_path = args.out or (args.log_path.rsplit("/", 1)[-1].replace(".eval", "") + "_spotcheck.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(chosen)} sampled transcripts -> {out_path}")


if __name__ == "__main__":
    main()
