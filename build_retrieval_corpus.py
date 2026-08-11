"""
Phase 6 setup: build the retrieval corpus and run the contamination check.

One-time (well, rerun-when-you-want-a-different-corpus) script -- NOT part
of the eval loop. Retrieval itself (BM25 index load + top-k lookup) lives
in elicit_task.py, wired into build_solver() via the `retrieval` toggle;
this script only produces the two artifacts that toggle depends on:
    - suites/retrieval_corpus.jsonl   (the passages)
    - suites/contamination_report.json (the check required before trusting them)

CORPUS CHOICE, v2 -- topical, via Wikipedia category membership (not a
random sample, see history below). v1 used a random sample of Wikipedia
specifically to dodge contamination risk, on the theory that a curated
"textbook-adjacent" corpus would make it easy to accidentally hand-pick
passages close to GPQA/MATH's own source material. That reasoning was
sound on contamination but wrong on the tradeoff: checked v1 against real
GPQA/MATH questions and it returned near-random, topically irrelevant
passages every time (e.g. "Toyota Group" for a quantum-mechanics
question) -- a corpus that's safe but useless isn't actually a good
tradeoff. v2 fixes relevance via Wikipedia's CATEGORY system (subject
labels on articles), not by looking at GPQA/MATH's actual question
content -- so it's not circular/contamination-inducing on its own, and
the exact same contamination check below (SHINGLE_SIZE-word shingle
overlap) is what actually verifies safety, same as v1. Topic categories
are deliberately SUBJECT-SPECIFIC (e.g. "Quantum mechanics", not just
"Physics") -- broad top-level categories mostly contain subcategories
like "Physicists" (biographies) and "Physics by country" (institutional
lists), not conceptual content, so starting subject-specific avoids that
drift without needing deep, hard-to-tune category recursion.

Fetches article TEXT directly from Wikipedia's own API
(action=query&prop=extracts), not by streaming/filtering the 6.4M-article
HF wikimedia/wikipedia dump by title -- confirmed the extracts endpoint
returns clean plain-text article bodies directly for a batch of titles,
which avoids having to scan a large fraction of the full HF dump looking
for a comparatively small target title set.

Each doc is truncated to its first ~1500 characters (a passage, not the
full article) -- long enough for real content, short enough to keep the
BM25 index and per-eval prompt injection cheap.

CONTAMINATION CHECK: builds the set of all 8-word shingles across the
entire corpus, then checks every GPQA/MATH/HumanEval question+target for
shingle overlap against that set. Shingle-set membership (not pairwise
string comparison) so this is O(corpus + eval), not O(corpus x eval) --
tractable at this corpus size, and would still be tractable at 10x. Any
hit is a real thing to go look at by hand, not an automatic fail (could
be an extremely generic 8-word phrase, e.g. a numbered-list coincidence
-- see _is_generic_numeric below); reports every hit for that manual
follow-up rather than silently thresholding them away.

Usage:
    python build_retrieval_corpus.py
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()  # GPQA (used in the contamination check) is HF-gated and
                # needs HF_TOKEN -- not auto-loaded outside Inspect's own
                # eval machinery, unlike when running via `inspect eval`.

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "elicitation-harness-research/1.0"}

# Subject-specific, not top-level -- see rationale in the module docstring.
# Grouped by which suite each is meant to help; kept as one flat corpus
# though (retrieval doesn't need to know which suite is asking).
CATEGORIES = [
    # GPQA: physics
    "Category:Quantum mechanics", "Category:Classical mechanics",
    "Category:Thermodynamics", "Category:Electromagnetism", "Category:Optics",
    "Category:Particle physics", "Category:Nuclear physics",
    "Category:Special relativity", "Category:Statistical mechanics",
    # GPQA: chemistry
    "Category:Organic chemistry", "Category:Physical chemistry",
    "Category:Inorganic chemistry", "Category:Chemical bonding",
    "Category:Chemical reactions", "Category:Stereochemistry",
    "Category:Analytical chemistry",
    # GPQA: biology
    "Category:Molecular biology", "Category:Genetics", "Category:Cell biology",
    "Category:Biochemistry", "Category:Enzymes", "Category:Evolutionary biology",
    "Category:Microbiology",
    # MATH (intermediate_algebra, plus adjacent math subjects)
    "Category:Elementary algebra", "Category:Algebra",
    "Category:Polynomials", "Category:Equations", "Category:Number theory",
    "Category:Mathematical analysis", "Category:Inequalities",
    "Category:Functions and mappings", "Category:Sequences and series",
]

# Subcategory titles containing any of these are excluded when descending
# one level deeper -- biographical/institutional/meta content, not concepts.
SUBCAT_EXCLUDE_PATTERNS = [
    "people", "physicists", "chemists", "biologists", "mathematicians",
    "by country", "by year", "by nationality", "history of", "list of",
    "lists of", "works about", "in fiction", "awards", "societies",
    "organizations", "journals", "conferences", "universities", "births",
    "deaths", "stubs", "wikipedia", "categories",
]

MAX_TITLES = 4000       # cap on total unique article titles collected
PASSAGE_CHARS = 1500
SHINGLE_SIZE = 8  # words


def _api_get(params: dict) -> dict:
    """Every Wikipedia API call goes through here -- centralized so the
    rate-limit handling below covers category traversal AND extract
    fetching, not just the latter (the first version only paced the
    extracts loop and immediately hit a 429 partway through the very
    first category's subcategory scan, since category traversal fans out
    into far more requests than the extracts loop does)."""
    for attempt in range(6):
        r = httpx.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 * (attempt + 1)))
            print(f"    (429, waiting {wait:.0f}s)")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(0.15)  # steady pacing, not just reactive backoff
        return r.json()
    raise RuntimeError(f"Wikipedia API: gave up after repeated 429s for {params}")


def _category_members(category: str, cmtype: str) -> list[dict]:
    """One category's direct members (articles or subcats), paginated."""
    members, cmcontinue = [], None
    while True:
        params = {
            "action": "query", "list": "categorymembers", "cmtitle": category,
            "cmlimit": 500, "cmtype": cmtype, "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _api_get(params)
        members.extend(data.get("query", {}).get("categorymembers", []))
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            return members


def collect_titles() -> list[str]:
    """BFS, depth 1: direct articles in each root category, plus direct
    articles in each root's non-excluded subcategories. Depth capped at 1
    deliberately -- deeper recursion drifts further from the subject-
    specific root with each hop and gets harder to keep on-topic without
    per-hop tuning; depth 1 already reaches thousands of candidate titles
    (see MAX_TITLES), which is enough to test relevance meaningfully."""
    titles: set[str] = set()
    for cat in CATEGORIES:
        print(f"  {cat}...")
        pages = _category_members(cat, "page")
        titles.update(p["title"] for p in pages)

        subcats = _category_members(cat, "subcat")
        subcats = [
            s for s in subcats
            if not any(pat in s["title"].lower() for pat in SUBCAT_EXCLUDE_PATTERNS)
        ]
        for sub in subcats:
            sub_pages = _category_members(sub["title"], "page")
            titles.update(p["title"] for p in sub_pages)

        if len(titles) >= MAX_TITLES:
            print(f"  hit MAX_TITLES={MAX_TITLES}, stopping category scan early")
            break

    return sorted(titles)[:MAX_TITLES]


def fetch_extracts(titles: list[str], batch_size: int = 20) -> list[dict]:
    """Wikipedia's extracts API caps how many titles you can batch per
    request (conservatively using 20 here); one request per batch. Pacing
    and 429 handling both live in _api_get, not here."""
    docs = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        # exintro=1 (intro section only, not the full article) is what
        # actually makes batching work -- the API silently caps a FULL
        # explaintext extract to exlimit=1 regardless of what's passed,
        # meaning the first version of this function was really only
        # fetching 1 real doc per 20-title batch and dropping the other
        # 19 as if they were redirects (they weren't -- verified by
        # re-querying one directly with exintro=1 and getting real
        # content back). An intro-only extract is still 400-2500+ chars
        # in practice, comfortably above PASSAGE_CHARS, so nothing here
        # is actually lost by not fetching the full article body.
        data = _api_get({
            "action": "query", "prop": "extracts", "titles": "|".join(batch),
            "explaintext": 1, "exintro": 1, "format": "json",
        })
        for pageid, page in data.get("query", {}).get("pages", {}).items():
            extract = page.get("extract", "")
            if not extract or pageid.startswith("-"):  # "-1" = missing page
                continue
            docs.append({
                "id": pageid,
                "title": page["title"],
                "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
                "passage": extract[:PASSAGE_CHARS],
            })
        if (i // batch_size) % 20 == 0:
            print(f"  fetched extracts for {i + len(batch)}/{len(titles)} titles")
    return docs


def build_corpus() -> list[dict]:
    print(f"Collecting article titles from {len(CATEGORIES)} subject categories...")
    titles = collect_titles()
    print(f"Collected {len(titles)} unique candidate titles.")

    print("Fetching article extracts...")
    docs = fetch_extracts(titles)
    print(f"Got {len(docs)} docs with non-empty extracts "
          f"({len(titles) - len(docs)} titles had no extract / were redirects).")
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
