"""
Elicitation task, generalized across task TYPES (plain Q&A, boxed-math,
code-execution), not just GSM8K-style question/answer.

IMPORTANT -- equal footing is a comparison rule, not a dataset-loading rule:
  - Comparing config A vs config B, or model X vs model Y, is only valid
    when both run on the IDENTICAL task suite (same dataset, same subject/
    split, same sample order -- every adapter below sets shuffle=False).
  - Supporting multiple task TYPES does not violate that. It lets you ask
    a *different* question: does the same scaffold component matter
    differently on different kinds of tasks (e.g. does tool_use help a lot
    on code but barely at all on pure math)? That's one of the project's
    headline claims and requires multiple task types to even test.
  - What you must NEVER do: pool or directly compare a raw score from one
    suite against a raw score from a different suite (e.g. "MATH accuracy
    0.55 vs SWE-bench accuracy 0.30, so MATH is easier"). Different suites
    use different scorers and difficulty scales -- report each suite's
    results separately, and only compare *deltas within* each suite
    (e.g. "tool_use added +12pts on SWE-bench vs +1pt on MATH").

Architecture: each dataset type is a small "adapter" -- just a loader
function and a scorer -- registered by name. The solver-building logic
(the six scaffold-component toggles) is completely shared and never
duplicated per adapter. Adding a new dataset type means writing one new
adapter, not touching anything else.

Run a single config directly:
    inspect eval elicit_task.py --model openai/gpt-4o-mini \\
        -T suite=math -T cot=true --limit 20
Or sweep via run_ablation.py, which now also loops over SUITES.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

# json, lru_cache: only needed by the retrieval component, commented out
# below (see "Retrieval component" block) -- re-enable both if retrieval
# comes back.
# import json
# from functools import lru_cache

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.solver import (
    Generate,
    Solver,
    chain_of_thought,
    generate,
    self_critique,
    solver,
    system_message,
    use_tools,
    TaskState,
)
from inspect_ai.tool import python
from inspect_ai.util import sandbox

# --------------------------------------------------------------------------
# Retrieval component -- DROPPED FOR NOW (2026-08-11), not deleted.
# Scope decision: doing 5 of 6 README components + Shapley + cross-model
# (Qwen/Llama) rather than all 6. Retrieval was the most troubled
# component this session (three real bugs: a random corpus that returned
# near-random passages on real GPQA/MATH questions, a Wikipedia rate
# limit, and a MediaWiki API quirk that silently discarded 95% of fetched
# articles) and even after the v2 category-filtered rebuild finished
# (suites/retrieval_corpus.jsonl, 3916 docs), its relevance was never
# smoke-tested. See process_log.md for the full history. Commented out,
# not deleted, so it's cheap to re-enable if there's time/budget later --
# uncomment this block, the `use_retrieval`/`retrieval` toggle lines in
# build_solver() and elicit() below, and the json/lru_cache imports above.
# --------------------------------------------------------------------------

# RETRIEVAL_CORPUS_PATH = "suites/retrieval_corpus.jsonl"
# RETRIEVAL_TOP_K = 3
#
#
# @lru_cache(maxsize=1)
# def _load_bm25_index():
#     """Lazy + cached: only paid once per process, and only if `retrieval`
#     is actually toggled on -- configs that never use retrieval never pay
#     for loading or indexing the corpus."""
#     from rank_bm25 import BM25Okapi
#
#     docs = []
#     with open(RETRIEVAL_CORPUS_PATH) as f:
#         for line in f:
#             docs.append(json.loads(line))
#     tokenized = [d["passage"].lower().split() for d in docs]
#     return BM25Okapi(tokenized), docs
#
#
# def _retrieve(query: str, k: int = RETRIEVAL_TOP_K) -> list[dict]:
#     bm25, docs = _load_bm25_index()
#     scores = bm25.get_scores(query.lower().split())
#     top_idx = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
#     return [docs[i] for i in top_idx]
#
#
# @solver
# def retrieval() -> Solver:
#     """Retrieves top-k BM25 passages for the sample's own input text and
#     prepends them to the user prompt, clearly labeled as reference
#     material rather than instructions -- this is the `retrieval` toggle's
#     entire job; everything else about answering the question is
#     unaffected. Runs BEFORE generate() in build_solver() so the retrieved
#     context is present for the model's first (and only, if cot/critique
#     are off) generation."""
#
#     async def solve(state: TaskState, generate: Generate) -> TaskState:
#         passages = _retrieve(state.input_text)
#         context = "\n\n".join(
#             f"[Reference {i+1}: {p['title']}]\n{p['passage']}"
#             for i, p in enumerate(passages)
#         )
#         state.user_prompt.text = (
#             f"Reference material (may or may not be relevant to the "
#             f"question below):\n\n{context}\n\n---\n\n{state.user_prompt.text}"
#         )
#         return state
#
#     return solve


PLANNING_PROMPT = (
    "Before answering, break this problem down into a short numbered "
    "list of concrete steps needed to solve it. Output ONLY the "
    "numbered plan -- do not solve the problem yet."
)

# Deliberately does NOT hardcode an answer format (no "ANSWER: X", no
# "\boxed{}") -- planning() is suite-agnostic, and each suite's own
# system prompt (GPQA_SYSTEM_BASE, MATH_SYSTEM, HUMANEVAL_SYSTEM) already
# specifies its own format. This just points back at those instructions.
PLANNING_FOLLOWUP_PROMPT = (
    "Now answer the original question, following the plan above and the "
    "answer-format instructions from the system message."
)


@solver
def planning() -> Solver:
    """Decompose-then-execute: a genuinely separate generation turn that
    asks for a numbered plan FIRST, before the main answer -- not just
    chain_of_thought() under a different name (which reasons and answers
    in a single pass with no distinct, inspectable planning artifact).

    Implementation choice, and why: the plan is added as a REAL prior
    conversation turn (append a user message asking for a plan, call
    generate() to get it, which appends the assistant's plan response to
    state.messages) rather than manually splicing plan text into the
    prompt the way retrieval() does. This matters for Phase 7's own smoke
    test criteria: with the plan as a real turn, a transcript reviewer can
    directly see (a) whether the plan differs per question by reading the
    assistant's plan-turn response, and (b) whether the final answer
    actually references the plan, since it's genuinely earlier
    conversation history the model has access to during the main
    generate() step that follows -- not an inference from prompt text
    that was never actually part of the exchange.

    BUG FOUND AND FIXED (2026-08-11): the first version stopped right
    after the plan turn and relied on build_solver()'s own trailing
    generate() call to produce the final answer. That doesn't work --
    with the conversation ending on the model's OWN plan turn and no
    explicit next instruction, the model has nothing telling it to stop
    planning and start answering, and it just repeats/continues the plan
    instead. Confirmed via real transcripts: GPQA accuracy collapsed to
    0.04-0.12 (from a 0.40-0.48 bare baseline) with EVERY sample failing
    to produce a parseable ANSWER: line -- not a "planning hurts
    reasoning" finding, a missing prompt. Fixed by appending an explicit
    follow-up instruction (PLANNING_FOLLOWUP_PROMPT) after the plan,
    telling the model to now actually answer using the plan -- mirrors
    how self_critique()'s own completion_template re-poses the question
    and explicitly asks for a new answer, rather than trusting the model
    to infer that on its own."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.append(ChatMessageUser(content=PLANNING_PROMPT))
        state = await generate(state)
        state.messages.append(ChatMessageUser(content=PLANNING_FOLLOWUP_PROMPT))
        return state

    return solve


@solver
def multi_critic(
    critic_model: str | None = None, critic_config: GenerateConfig | None = None,
) -> Solver:
    """`multi_critic`: a critic model reviews and revises the primary
    model's answer -- Inspect's own `self_critique()` already supports an
    alternate `model=` for exactly this (no need to hand-roll a
    cross-model critique loop), so this is a thin, documented wrapper
    around it rather than new solver logic.

    `critic_model=None` (default) means the SAME model critiques its own
    answer -- behaviorally identical to the existing `critique` toggle.
    This is deliberate, not a missing feature: Phase 9's own acceptance
    criteria require comparing same-model-critic vs different-model-critic
    behavior (the README names judge-benchmark circularity as a real risk
    for this component), and having `multi_critic(critic_model=None)` and
    `multi_critic(critic_model="together/...")` both be valid, directly
    comparable calls is what makes that comparison possible -- it's a
    parameter change, not two different code paths that might not be
    apples-to-apples. For the actual 16-config ablation sweep, the
    critic model is set to "the other model in the pair" (see
    CRITIC_MODEL in run_ablation.py) so `multi_critic` is a genuine
    cross-model check there, not accidentally a duplicate of `critique`.

    Note: the default completion template asks for "ANSWER: $ANSWER" on
    its own line -- matches GPQA's letter_match scorer (the suite this
    study actually targets); would need a custom completion_template to
    fit MATH's \\boxed{} format if this ever runs there instead.

    `critic_config`: generation config for the critic model specifically.
    `self_critique(model=critic_model)` resolves a bare model-name string
    via `get_model(critic_model)` with NO config -- meaning it silently
    ignores whatever max_tokens/extra_body the eval-level `inspect_eval()`
    call set for the PRIMARY model, and falls back to that critic model's
    raw server default. Found the hard way: gemma's multi_critic runs
    against Qwen3-32B-as-critic hit the exact same max_tokens=65536-
    exceeds-40960-context error that had already been fixed for Qwen3-32B
    as a primary model (see process_log.md's "switch to DeepInfra" entry)
    -- the fix never propagated to the critic-model path because it's a
    separate, unconfigured get_model() call.

    This function is itself `@solver`-decorated (unlike the version that
    shipped originally) specifically so the `get_model(critic_model,
    config=critic_config)` call below happens LAZILY, inside `solve()`, at
    actual generation time -- not eagerly when `multi_critic()` is called
    while `build_solver()` constructs the Task. Resolving it eagerly (the
    first attempt at this fix) broke immediately: `elicit(...)` runs
    before `inspect_eval()` has had a chance to load `.env`, so
    `get_model()` raised `PrerequisiteError: No DEEPINFRA_API_KEY defined`
    on the very first call. `self_critique()`'s own `model` resolution
    already happens lazily inside its `solve()` for exactly this reason;
    this wrapper has to match that, not shortcut it."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        critic = get_model(critic_model, config=critic_config) if critic_model else None
        inner = self_critique(model=critic)
        return await inner(state, generate)

    return solve


# Number of byte-identical consecutive tool calls that triggers an early
# stop, and the outer round cap (belt-and-suspenders alongside
# run_shapley_sweep.py's working_limit -- this is meant to fire first,
# much earlier, for the specific failure mode it targets).
MAX_IDENTICAL_TOOL_CALLS = 3
MAX_TOOL_ROUNDS = 50


@solver
def generate_with_repeat_guard() -> Solver:
    """generate() with tool_calls="loop" (Inspect's default), except it
    detects a model stuck repeating the IDENTICAL tool call turn after
    turn and stops early instead of burning the full working_limit
    budget on it.

    Found via direct transcript inspection (2026-08-23, see
    process_log.md): tool_use+planning samples were burning 15-40x the
    token budget of every other config on gemma-3-27b-it -- up to 41M
    tokens for one 250-sample run, which is also what was driving
    repeated DeepInfra balance exhaustion. Root cause, confirmed by
    reading the actual repeated calls in the worst samples: NOT the
    model struggling with a hard problem. Most had already computed a
    correct \\boxed{...} answer and tried to "submit" it by calling
    print('\\boxed{N}') through the python() tool instead of writing it
    in their own reply. boxed_match only reads the assistant's own text
    content, so that's invisible to the scorer; the tool's output never
    changes in response, and the model just repeats the exact same call
    (500+ times in the worst observed cases) until working_limit finally
    cuts it off at 600s.

    This stops that specific degenerate loop far earlier -- after
    MAX_IDENTICAL_TOOL_CALLS consecutive byte-identical calls -- without
    touching what a genuinely varied, productive tool_use conversation is
    allowed to do: a model trying different code across turns is never
    affected by this, only exact repeats are. Ends the sample the same
    way working_limit already does (whatever text exists gets scored,
    typically wrong for a sample that never emitted a visible answer) --
    just at a small fraction of the token cost."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        last_sig: tuple[tuple[str, str], ...] | None = None
        repeat_count = 0
        for _ in range(MAX_TOOL_ROUNDS):
            state = await generate(state, tool_calls="single")
            last_assistant = next(
                (m for m in reversed(state.messages) if m.role == "assistant"),
                None,
            )
            tool_calls = getattr(last_assistant, "tool_calls", None)
            if not tool_calls:
                break  # model answered directly, no more tool calls -- done
            sig = tuple(
                (c.function, json.dumps(c.arguments, sort_keys=True))
                for c in tool_calls
            )
            repeat_count = repeat_count + 1 if sig == last_sig else 1
            last_sig = sig
            if repeat_count >= MAX_IDENTICAL_TOOL_CALLS:
                break
        return state

    return solve


# --------------------------------------------------------------------------
# Shared solver-building logic (identical across every task type)
# --------------------------------------------------------------------------

def build_solver(
    cot: bool, critique: bool, system_prompt: str,
    tool_use: bool = False, use_retrieval: bool = False, use_planning: bool = False,
    use_multi_critic: bool = False, critic_model: str | None = None,
    critic_config: GenerateConfig | None = None,
):
    steps = [system_message(system_prompt)]
    # retrieval dropped for now -- see "Retrieval component" comment block
    # above. use_retrieval is accepted but currently a no-op; uncomment
    # below (and the retrieval() solver above) to bring it back.
    # if use_retrieval:
    #     steps.append(retrieval())
    if use_planning:
        steps.append(planning())
    if cot:
        steps.append(chain_of_thought())
    if tool_use:
        # timeout=30 mirrors the guard already used for humaneval's own
        # sandboxed code_execution_match (see that scorer's comment) --
        # found the hard way this component needed it too: a MATH pilot
        # run hit a real infinite loop inside the sandbox that ran
        # unbounded for 9+ hours before being noticed and killed
        # manually (see process_log.md's "runaway tool_use process"
        # entry). python()'s own timeout kwarg was always there; it was
        # just never passed.
        steps.append(use_tools(python(timeout=30)))
        # generate_with_repeat_guard(), not plain generate(): see that
        # solver's docstring for why tool_use specifically needs its own
        # loop instead of Inspect's default tool_calls="loop" generate().
        steps.append(generate_with_repeat_guard())
    else:
        steps.append(generate())
    if critique:
        steps.append(self_critique())
    if use_multi_critic:
        steps.append(multi_critic(critic_model, critic_config))
    return steps


# --------------------------------------------------------------------------
# Adapter interface: every dataset type provides a loader + a scorer +
# a system prompt tailored to its answer format. Nothing else varies.
# `sandbox` is None for every adapter except code-execution ones, which
# need sandbox="docker" on the Task for both the tool_use python() tool
# and code_execution_match's own test-running to actually work.
# --------------------------------------------------------------------------

@dataclass
class TaskAdapter:
    name: str
    system_prompt: str
    load: Callable[[], Dataset]
    scorer: Callable[[], Scorer]
    sandbox: str | None = None


# ---- Adapter: plain numeric Q&A (GSM8K-style) -----------------------------

def _load_gsm8k() -> Dataset:
    def record_to_sample(record):
        target = record["answer"].split("####")[-1].strip().replace(",", "")
        return Sample(input=record["question"], target=target)

    return hf_dataset(
        "gsm8k", name="main", split="test",
        sample_fields=record_to_sample, shuffle=False,
    )


GSM8K_SYSTEM = (
    "Solve the math problem. Show brief reasoning, then write the final "
    "answer as a plain number on its own line at the end."
)


# ---- Adapter: boxed-answer math (Hendrycks MATH-style) --------------------

# "algebra" was the original choice but bare gpt-4o-mini scores 0.909 on it
# at n=1000 -- a ceiling problem, same failure mode that rejected GSM8K.
# Piloted all 7 Hendrycks MATH subjects (n=40, then n=500 on the top two):
# intermediate_algebra and geometry both land at ~0.55 bare accuracy at
# n=500 (statistically tied), but geometry's incorrect cases showed a real,
# recurring false-negative pattern (\pi has no symbolic handling anywhere
# in _sympy_equivalent -- \frac{9}{2}\pi vs \frac{9\pi}{2} etc.) that
# intermediate_algebra's did not. Picked intermediate_algebra: same
# headroom, no outstanding scorer gap, and a larger total pool (903 vs
# geometry's hard-capped 479) if a bigger sample is ever needed.
MATH_SUBJECT = "intermediate_algebra"

# Was "Show your reasoning, then give the final answer..." unconditionally
# -- used for every config regardless of the `cot` toggle, so the "bare"
# MATH condition was never actually reasoning-free the way GPQA's is.
# GPQA hit the identical bug earlier (see the CoT-leak fix in
# process_log.md) and the fix there was to make the FIXED system prompt
# bare-safe ("do not explain your reasoning") and rely entirely on the
# `chain_of_thought()` solver step -- added only when cot=True in
# build_solver() above -- as the sole source of reasoning. Mirrored here.
MATH_SYSTEM = (
    "Solve the math problem. Give ONLY the final answer, wrapped in "
    "\\boxed{...}, on its own line. Do not show your work or explain "
    "your reasoning."
)


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def _consume_balanced(s: str, i: int) -> tuple[str, int] | None:
    """s[i] must be '{'. Return (substring including braces, index just
    past the matching closing brace), correctly handling nested braces
    (e.g. the inner "{6}" inside the outer "{\\sqrt{6}}")."""
    if i >= len(s) or s[i] != "{":
        return None
    depth, j = 1, i + 1
    while j < len(s) and depth > 0:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return (s[i:j], j) if depth == 0 else None


def _consume_frac_arg(s: str, i: int) -> tuple[str, int] | None:
    """Consume one \\frac/\\sqrt argument starting at s[i]: a {...} group
    (nesting allowed), a \\command optionally followed by its own {...}
    group (e.g. \\sqrt{7}), or a single bare character (a digit, in
    practice)."""
    if i >= len(s):
        return None
    if s[i] == "{":
        return _consume_balanced(s, i)
    if s[i] == "\\":
        j = i + 1
        while j < len(s) and s[j].isalpha():
            j += 1
        if j == i + 1:
            return None
        if j < len(s) and s[j] == "{":
            grp = _consume_balanced(s, j)
            return (s[i:j] + grp[0], grp[1]) if grp else None
        return (s[i:j], j)
    return (s[i], i + 1)


def _brace_frac_sqrt_args(s: str) -> str:
    """Normalize brace-less \\frac / \\sqrt shorthand into the fully-braced
    canonical form: \\frac56 / \\frac 56 / \\frac9{5} -> \\frac{9}{5},
    \\sqrt7 / \\sqrt 7 -> \\sqrt{7}. Found in real MATH transcripts (both
    styles are used interchangeably by the dataset and are never
    meaningfully different) -- without this, two answers that are
    byte-for-byte the same once rendered were failing both the string and
    sympy comparison because the sympy fallback only recognizes the
    fully-braced form.

    Uses an explicit balanced-brace scanner (_consume_frac_arg), not a
    single regex, because a regex requiring "{[^{}]*}" for an "already
    braced" arg silently fails to match -- and therefore skips the WHOLE
    \\frac, including a brace-less SIBLING arg that still needed fixing --
    the moment either arg has any nesting in it. Real example that broke
    the old regex version: \\frac{\\sqrt6}3, where the numerator itself
    contains \\sqrt6."""
    # \sqrt7 / \sqrt 7 -> \sqrt{7} first, so a nested \sqrt inside a \frac
    # arg is already canonical by the time the \frac scan below reads it.
    s = re.sub(r"\\sqrt(?!\{)\s*(\d)", r"\\sqrt{\1}", s)

    out, i, n = [], 0, len(s)
    while i < n:
        if s.startswith("\\frac", i):
            j = i + 5
            while j < n and s[j] == " ":
                j += 1
            first = _consume_frac_arg(s, j)
            if first is not None:
                a, j2 = first
                while j2 < n and s[j2] == " ":
                    j2 += 1
                second = _consume_frac_arg(s, j2)
                if second is not None:
                    b, j3 = second
                    a = a if a.startswith("{") else "{%s}" % a
                    b = b if b.startswith("{") else "{%s}" % b
                    out.append("\\frac%s%s" % (a, b))
                    i = j3
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _strip_text_wrapper(s: str) -> str:
    """\\text{...} shows up in MATH answers two unrelated ways: wrapping a
    non-numeric answer entirely (\\text{Evelyn}, \\text{June 20}) or tacked
    on as a discardable unit/label suffix on a numeric answer
    (118 \\text{ dollars}, 440\\text{ cm}^2 -- the trailing ^2 there is
    part of the "cm^2" unit, not a real exponent, so it has to be stripped
    along with the \\text{} it's attached to, or a bare "^2" would be left
    behind corrupting the number). Disambiguate the two cases by whether
    \\text{...} is the ENTIRE answer (unwrap, keep the contents) or just
    part of it (strip it as a label, discard the contents)."""
    stripped = s.strip()
    m = re.fullmatch(r"\\text\{([^{}]*)\}", stripped)
    if m:
        return m.group(1)
    return re.sub(r"\\text\{[^{}]*\}(\^\{?\d+\}?)?", "", s)


def _clean_latex(ans: str) -> str:
    """Normalize superficial LaTeX/formatting differences that don't change
    meaning: \\dfrac/\\tfrac -> \\frac, brace-less \\frac/\\sqrt shorthand
    braced, \\text{} unwrapped or stripped, \\left \\right removed,
    "+\\infty" treated as "\\infty", spacing commands (incl. "\\ ")
    removed, a leading "x \\in " / "x = " variable prefix stripped, a
    literal \\$ (currency) stripped, thousands-separator commas collapsed
    when the whole answer is nothing but a grouped number, whitespace
    collapsed, trailing period and $ removed."""
    s = ans.strip()
    s = re.sub(r"\\[dt]?frac", r"\\frac", s)  # \dfrac, \tfrac -> \frac
    # ^ was "\\d?frac", which only ever made the "d" optional -- despite
    # this docstring (and the old one) claiming \tfrac was handled, it
    # silently never matched. Found via a fresh \frac{10}{3} vs
    # \tfrac{10}{3} false negative that had nothing to do with sympy at all.
    s = _brace_frac_sqrt_args(s)
    s = _strip_text_wrapper(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("+\\infty", "\\infty")  # "+infinity" == "infinity"
    s = re.sub(r"\\[,;:! ]", "", s)                 # LaTeX spacing commands
    s = re.sub(r"^[a-zA-Z]\s*(\\in|=)\s*", "", s)   # "x \in ", "x = " prefix
    # NOTE: strip "\$" (escaped currency dollar) before the bare "$" strip
    # below -- doing only the bare strip turns "\$40" into the broken "\40"
    # (dangling backslash) instead of "40", which was a real false negative.
    s = s.replace("\\$", "").replace("$", "").rstrip(".").strip()
    # "$15,000" typed as "15,000": collapse thousands-separator commas, but
    # ONLY when the whole remaining answer is nothing but a comma-grouped
    # number -- a genuine multi-value list ("4,6,14,15") must never be
    # touched. Residual, documented ambiguity: a list of three three-digit
    # values ("100,200,300") is indistinguishable from one grouped number
    # by string shape alone; no such case has been observed in this
    # dataset so far.
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    return s


def _sympy_equivalent(a: str, b: str) -> bool | None:
    """Try to parse both answers as math expressions and check exact
    symbolic/numeric equivalence (catches \\frac{9}{7} == 9/7 == 1.2857...,
    \\frac{11}{2} == 5.5, 7(x-3)(x+3) == 7(x+3)(x-3), x^{9} == x^9, ordered
    pairs like (1, 4.5) == (1, \\frac{9}{2}), etc). Returns None if sympy
    isn't installed or either side fails to parse -- caller should fall
    back to string match in that case, not treat None as "not equal"."""
    try:
        from sympy import sympify, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
    except ImportError:
        return None

    # Plain sympify() rejects "7(x-3)(x+3)" (implicit multiplication) and
    # treats bare "^" as XOR rather than power -- both are valid MATH answer
    # styles. This transform set is sympy's own tolerant parser for exactly
    # that, used as a last-resort tier below (never as the first attempt,
    # since a more permissive parser is more likely to mis-parse ambiguous
    # input into *something* rather than failing loudly).
    TOLERANT = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    def _latex_to_sympy_str(s: str) -> str:
        """Manual \\frac{a}{b} -> (a)/(b), \\sqrt{x} -> sqrt(x), x^{9} ->
        x^(9) conversion, used when the optional antlr4 LaTeX parser isn't
        installed. Handles one level of nesting, which covers the vast
        majority of MATH dataset answers (e.g. \\frac{11}{2},
        \\frac{1}{\\sqrt{2}})."""
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
            s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        # x^{9} -> x^(9); braces aren't valid here for convert_xor/parse_expr
        # (Python reads "{9}" as a set literal, not a grouped exponent).
        s = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", s)
        return s

    def _parse(s: str):
        # try the proper LaTeX parser first, if antlr4 is installed
        try:
            return parse_latex(s)
        except Exception:
            pass
        # fall back to plain sympy syntax (handles "9/7", "5.5", "2*x")
        try:
            return sympify(s)
        except Exception:
            pass
        # manually rewrite common LaTeX macros, then sympify
        rewritten = _latex_to_sympy_str(s)
        try:
            return sympify(rewritten)
        except Exception:
            pass
        # last resort: sympy's tolerant parser, for implicit multiplication
        # ("(x-3)(x+3)") and bare "^" exponents plain sympify() rejects
        try:
            return parse_expr(rewritten, transformations=TOLERANT)
        except Exception:
            return None

    ea, eb = _parse(a), _parse(b)
    if ea is None or eb is None:
        return None
    # Ordered pairs / tuples (coordinate-style answers): sympify() returns
    # a plain Python tuple for "(1, 4.5)", which doesn't support "-", so it
    # has to be compared element-wise rather than via simplify(ea - eb).
    if isinstance(ea, tuple) and isinstance(eb, tuple):
        if len(ea) != len(eb):
            return False
        try:
            return all(simplify(x - y) == 0 for x, y in zip(ea, eb))
        except Exception:
            return None
    try:
        return simplify(ea - eb) == 0
    except Exception:
        try:
            return ea.equals(eb) is True
        except Exception:
            return None


def _math_answers_match(model_answer: str, target: str) -> bool:
    """Tiered check: cheap string match first, then symbolic/numeric
    equivalence via sympy for anything that only differs in representation
    (fraction vs decimal, \\dfrac vs \\frac, an "x \\in " prefix, etc)."""
    a, b = _clean_latex(model_answer), _clean_latex(target)
    if a == b:
        return True
    result = _sympy_equivalent(a, b)
    if result is not None:
        return result
    # sympy unavailable or couldn't parse (e.g. it's an interval/set, not a
    # bare expression) -- one more pass, stripping remaining LaTeX wrappers
    # like \left[ \right] that _clean_latex already removes, then compare.
    return a == b


def _load_math() -> Dataset:
    def record_to_sample(record):
        target = _extract_boxed(record["solution"]) or ""
        return Sample(
            input=record["problem"], target=target,
            metadata={"level": record.get("level"), "type": record.get("type")},
        )

    return hf_dataset(
        "EleutherAI/hendrycks_math", name=MATH_SUBJECT, split="test",
        sample_fields=record_to_sample, shuffle=False,
    )


@scorer(metrics=[accuracy(), stderr()])
def boxed_match():
    async def score(state: TaskState, target: Target):
        model_answer = _extract_boxed(state.output.completion)
        correct = model_answer is not None and _math_answers_match(
            model_answer, target.text
        )
        return Score(
            value="C" if correct else "I",
            answer=model_answer or "(no \\boxed{} found)",
            explanation=(
                None if correct or model_answer is None
                else f"model gave '{model_answer}', target was '{target.text}'"
            ),
        )

    return score


# ---- Adapter: code execution (HumanEval-style) -----------------------------
# Needs a Docker sandbox to actually execute code safely -- wired via
# TaskAdapter's sandbox="docker" in the registry below, which elicit()
# passes straight to the Task. Both the tool_use python() tool and this
# adapter's own scorer depend on that sandbox existing; the "docker run
# hello-world" / Docker Desktop check is a real prerequisite, not optional.

def _load_humaneval() -> Dataset:
    def record_to_sample(record):
        return Sample(
            input=record["prompt"],
            target=record["test"],  # a pytest-style test string defining check(candidate)
            metadata={"entry_point": record["entry_point"]},
        )

    return hf_dataset(
        "openai/openai_humaneval", split="test",
        sample_fields=record_to_sample, shuffle=False,
    )


HUMANEVAL_SYSTEM = (
    "Complete the Python function. Return ONLY the full function "
    "(signature + body), no explanation and no markdown code fences."
)


def _strip_code_fence(text: str) -> str:
    """Strip a markdown ```python ... ``` (or bare ```) fence if the model
    wrapped its answer in one despite being told not to -- an extremely
    common chat-model habit, not a hypothetical edge case."""
    text = text.strip()
    m = re.match(r"^```(?:python)?\s*\n(.*?)\n?```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


@scorer(metrics=[accuracy(), stderr()])
def code_execution_match():
    """Executes the model's completed function against HumanEval's own
    check(candidate) test harness inside the task's sandbox (see
    sandbox="docker" on the humaneval adapter below) and scores pass/fail
    on the actual execution result -- not a text match, the whole point of
    this suite existing."""

    async def score(state: TaskState, target: Target):
        code = _strip_code_fence(state.output.completion)
        entry_point = state.metadata.get("entry_point")
        script = f"{code}\n\n{target.text}\n\ncheck({entry_point})\n"

        sb = sandbox()
        await sb.write_file("test_solution.py", script)
        try:
            result = await sb.exec(["python3", "test_solution.py"], timeout=30)
        except TimeoutError:
            return Score(
                value="I", answer=code,
                explanation="execution timed out after 30s (likely an infinite loop)",
            )

        correct = result.success
        return Score(
            value="C" if correct else "I",
            answer=code,
            explanation=None if correct else (result.stderr or result.stdout)[-1000:],
        )

    return score


# ---- Adapter: multiple choice (GPQA Diamond) -------------------------------
# The whole reason this adapter is worth having: scoring is a single-letter
# match. No LaTeX, no numeric-vs-symbolic equivalence, none of the false-
# negative headaches from the MATH adapter. Bare accuracy for gpt-4o-mini
# came in around 0.45 in the smoke test -- well below the ~0.70 human-
# expert ceiling, real headroom for scaffold components to move the needle.

GPQA_LETTERS = "ABCD"

GPQA_SYSTEM_BASE = (
    "Answer the multiple-choice question directly. Do not explain your "
    "reasoning. On its own line write your answer as exactly: "
    "ANSWER: X (where X is the letter of the correct choice)."
)

GPQA_SYSTEM = (
    "Answer the multiple-choice question. Reason through it, then on its "
    "own final line write your answer as exactly: ANSWER: X (where X is "
    "the letter of the correct choice)."
)


def _load_gpqa() -> Dataset:
    import random

    def record_to_sample(record):
        question = record["Question"]
        correct = record["Correct Answer"]
        options = [
            record["Incorrect Answer 1"],
            record["Incorrect Answer 2"],
            record["Incorrect Answer 3"],
            correct,
        ]
        # Shuffle option order so the correct answer isn't always in the
        # same position. Seed off the question text (not a global RNG) so
        # the shuffle is DETERMINISTIC and IDENTICAL every time this loads
        # -- required for equal footing: every config/model/seed run must
        # see the exact same option ordering for a given question, or a
        # score difference could just be "the letters got reshuffled."
        rng = random.Random(question)
        rng.shuffle(options)
        correct_letter = GPQA_LETTERS[options.index(correct)]

        formatted_choices = "\n".join(
            f"{letter}) {opt}" for letter, opt in zip(GPQA_LETTERS, options)
        )
        return Sample(
            input=f"{question}\n\n{formatted_choices}",
            target=correct_letter,
        )

    return hf_dataset(
        "Idavidrein/gpqa", name="gpqa_diamond", split="train",
        sample_fields=record_to_sample, shuffle=False,
    )


@scorer(metrics=[accuracy(), stderr()])
def letter_match():
    async def score(state: TaskState, target: Target):
        text = state.output.completion
        m = re.search(r"ANSWER:\s*\(?([A-D])\)?", text, re.IGNORECASE)
        model_letter = m.group(1).upper() if m else None
        correct = model_letter == target.text
        return Score(
            value="C" if correct else "I",
            answer=model_letter or "(no ANSWER: line found)",
            explanation=(
                None if correct
                else f"model said '{model_letter}', correct was '{target.text}'"
            ),
        )

    return score


# --------------------------------------------------------------------------
# Registry -- add a new dataset type by adding one entry here
# --------------------------------------------------------------------------

ADAPTERS: dict[str, TaskAdapter] = {
    "gsm8k": TaskAdapter("gsm8k", GSM8K_SYSTEM, _load_gsm8k, lambda: __import__(
        "inspect_ai.scorer", fromlist=["match"]
    ).match(location="end", numeric=True)),
    "math": TaskAdapter("math", MATH_SYSTEM, _load_math, boxed_match),
    "gpqa": TaskAdapter("gpqa", GPQA_SYSTEM_BASE, _load_gpqa, letter_match),
    "humaneval": TaskAdapter(
        "humaneval", HUMANEVAL_SYSTEM, _load_humaneval, code_execution_match,
        sandbox="docker",
    ),
}


@task
def elicit(
    suite: str = "math", cot: bool = False, critique: bool = False,
    tool_use: bool = False, retrieval: bool = False, planning: bool = False,
    multi_critic: bool = False, critic_model: str | None = None,
    critic_config: GenerateConfig | None = None,
):
    if suite not in ADAPTERS:
        raise ValueError(
            f"Unknown suite '{suite}'. Available: {list(ADAPTERS)}. "
            f"To add one, register a new TaskAdapter above."
        )
    adapter = ADAPTERS[suite]
    # tool_use needs an actual sandbox for python() to execute in, or
    # Inspect crashes the whole task the moment the model invokes the
    # tool (ProcessLookupError: no sandbox provided) -- found this the
    # hard way running tool_use on GPQA, which has no sandbox of its own
    # (only humaneval does). Only add docker when tool_use is actually
    # toggled on for THIS config, not unconditionally for every GPQA
    # config -- the other 15 configs in the 4-component sweep never
    # touch it and shouldn't pay Docker's per-sample startup cost.
    sandbox = adapter.sandbox or ("docker" if tool_use else None)
    return Task(
        dataset=adapter.load(),
        solver=build_solver(
            cot, critique, adapter.system_prompt, tool_use,
            use_retrieval=retrieval, use_planning=planning,
            use_multi_critic=multi_critic, critic_model=critic_model,
            critic_config=critic_config,
        ),
        scorer=adapter.scorer(),
        sandbox=sandbox,
    )
