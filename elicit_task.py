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

import re
from dataclasses import dataclass
from typing import Callable

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.solver import (
    chain_of_thought,
    generate,
    self_critique,
    system_message,
    use_tools,
    TaskState,
)
from inspect_ai.tool import python
from inspect_ai.util import sandbox

# --------------------------------------------------------------------------
# Shared solver-building logic (identical across every task type)
# --------------------------------------------------------------------------

def build_solver(cot: bool, critique: bool, system_prompt: str, tool_use: bool = False):
    steps = [system_message(system_prompt)]
    if cot:
        steps.append(chain_of_thought())
    if tool_use:
        # Inspect's generate() handles the tool-call/tool-result loop
        # automatically once a tool is registered via use_tools() -- no
        # extra looping logic needed here.
        steps.append(use_tools(python()))
    steps.append(generate())
    if critique:
        steps.append(self_critique())
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
    tool_use: bool = False,
):
    if suite not in ADAPTERS:
        raise ValueError(
            f"Unknown suite '{suite}'. Available: {list(ADAPTERS)}. "
            f"To add one, register a new TaskAdapter above."
        )
    adapter = ADAPTERS[suite]
    return Task(
        dataset=adapter.load(),
        solver=build_solver(cot, critique, adapter.system_prompt, tool_use),
        scorer=adapter.scorer(),
        sandbox=adapter.sandbox,
    )
