"""
Phase 6 setup: build the retrieval corpus and run the contamination check.

One-time (well, rerun-when-you-want-a-bigger-corpus) script -- NOT part of
the eval loop. Retrieval itself (BM25 index load + top-k lookup) lives in
elicit_task.py, wired into build_solver() via the `retrieval` toggle; this
script only produces the two artifacts that toggle depends on:
    - suites/retrieval_corpus.jsonl   (the passages)
    - suites/contamination_report.json (the check required before trusting them)

CORPUS CHOICE: a random sample of English Wikipedia (wikimedia/wikipedia,
20231101.en dump), not a curated science/math subset. Deliberate choice,
not laziness -- a curated "textbook-adjacent" corpus would make it easy to
accidentally hand-pick passages close to GPQA/MATH's own source material,
which is exactly the contamination risk this phase is supposed to guard
against. A general corpus is a more honest test of whether retrieval helps
at all, at the cost of a lower hit-rate for any specific question -- that
tradeoff gets checked empirically in the Phase 6 smoke test (spot-check
retrieved passages for topical relevance), not assumed here.

Sampling: streaming, shuffled with a fixed seed (reproducible), first
N_DOCS taken after the shuffle buffer fills. Each doc is truncated to its
first ~1500 characters (a passage, not the full article) -- long enough
for real content, short enough to keep the BM25 index and per-eval prompt
injection cheap.

CONTAMINATION CHECK: builds the set of all 8-word shingles across the
entire corpus, then checks every GPQA/MATH/HumanEval question+target for
shingle overlap against that set. Shingle-set membership (not pairwise
string comparison) so this is O(corpus + eval), not O(corpus x eval) --
tractable at this corpus size, and would still be tractable at 10x. Any
hit is a real thing to go look at by hand, not an automatic fail (could
be an extremely generic 8-word phrase); reports every hit for that manual
follow-up rather than silently thresholding them away.

Usage:
    python build_retrieval_corpus.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()  # GPQA (used in the contamination check) is HF-gated and
                # needs HF_TOKEN -- not auto-loaded outside Inspect's own
                # eval machinery, unlike when running via `inspect eval`.

N_DOCS = 5000
SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 10000
PASSAGE_CHARS = 1500
SHINGLE_SIZE = 8  # words


def build_corpus() -> list[dict]:
    print(f"Streaming wikimedia/wikipedia 20231101.en, shuffled (seed={SHUFFLE_SEED}, "
          f"buffer={SHUFFLE_BUFFER}), taking first {N_DOCS} after shuffle...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)

    docs = []
    for i, record in enumerate(ds):
        if i >= N_DOCS:
            break
        docs.append({
            "id": record["id"],
            "title": record["title"],
            "url": record["url"],
            "passage": record["text"][:PASSAGE_CHARS],
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{N_DOCS} docs collected")
    return docs


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    w = _words(text)
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)} if len(w) >= k else set()


def _is_generic_numeric(shingle: str) -> bool:
    """A shingle that's entirely digits with no real words carries no
    content signal -- an 8-in-a-row match on "1 2 3 4 5 6 7 8" between a
    HumanEval test case and an unrelated sports-standings table is not
    contamination, it's an artifact of numbered lists being common
    everywhere. Filtered out of the pass/fail verdict, but NOT silently
    dropped from the report -- every hit, including these, stays listed
    so this filter itself stays auditable rather than hiding a real
    finding behind a heuristic."""
    return all(w.isdigit() for w in shingle.split())


def check_contamination(corpus: list[dict]) -> dict:
    """Shingle-overlap scan between the retrieval corpus and every
    GPQA/MATH/HumanEval question+target. Returns a report dict, always
    written to disk regardless of outcome -- a clean result still needs
    to be on record as a check that was actually run, per Phase 3/6's
    "write down how this was checked, not just an assertion" standard."""
    print("\nBuilding corpus shingle index for contamination check...")
    corpus_shingles: dict[str, list[str]] = defaultdict(list)  # shingle -> [doc titles]
    for doc in corpus:
        for sh in _shingles(doc["passage"]):
            corpus_shingles[sh].append(doc["title"])

    import elicit_task

    hits = []
    checked = 0
    for suite_name, adapter in [
        ("gpqa", elicit_task.ADAPTERS["gpqa"]),
        ("math", elicit_task.ADAPTERS["math"]),
        ("humaneval", elicit_task.ADAPTERS["humaneval"]),
    ]:
        print(f"  checking {suite_name}...")
        dataset = adapter.load()
        for sample in dataset:
            checked += 1
            text = f"{sample.input}\n{sample.target}"
            for sh in _shingles(text):
                if sh in corpus_shingles:
                    hits.append({
                        "suite": suite_name,
                        "sample_id": str(sample.id),
                        "shingle": sh,
                        "matched_corpus_docs": corpus_shingles[sh],
                    })

    substantive_hits = [h for h in hits if not _is_generic_numeric(h["shingle"])]
    generic_numeric_hits = [h for h in hits if _is_generic_numeric(h["shingle"])]

    if not substantive_hits:
        verdict = "CLEAN -- zero substantive shingle overlap"
        if generic_numeric_hits:
            verdict += (f" ({len(generic_numeric_hits)} generic all-digit "
                        f"shingle(s) found and excluded, e.g. numbered-list "
                        f"coincidences like '1 2 3 4 5 6 7 8' -- see "
                        f"generic_numeric_hits below, not a contamination risk)")
    else:
        verdict = f"{len(substantive_hits)} substantive shingle overlap(s) -- manual review required"

    report = {
        "corpus_size": len(corpus),
        "shingle_size_words": SHINGLE_SIZE,
        "eval_samples_checked": checked,
        "suites_checked": ["gpqa", "math", "humaneval"],
        "substantive_hits": substantive_hits,
        "generic_numeric_hits": generic_numeric_hits,
        "verdict": verdict,
    }
    return report


def main():
    Path("suites").mkdir(exist_ok=True)
    corpus_path = Path("suites/retrieval_corpus.jsonl")

    if corpus_path.exists():
        print(f"{corpus_path} already exists, reusing it (delete it first "
              f"to regenerate from scratch).")
        with open(corpus_path) as f:
            corpus = [json.loads(line) for line in f]
    else:
        corpus = build_corpus()
        with open(corpus_path, "w") as f:
            for doc in corpus:
                f.write(json.dumps(doc) + "\n")
        print(f"\nWrote {len(corpus)} docs -> {corpus_path}")

    report = check_contamination(corpus)
    report_path = Path("suites/contamination_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nContamination check: {report['verdict']}")
    print(f"({report['eval_samples_checked']} eval samples checked across "
          f"{report['corpus_size']} corpus docs)")
    print(f"Full report -> {report_path}")


if __name__ == "__main__":
    main()
