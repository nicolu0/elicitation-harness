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
    TaskState,
)

# --------------------------------------------------------------------------
# Shared solver-building logic (identical across every task type)
# --------------------------------------------------------------------------

def build_solver(cot: bool, critique: bool, system_prompt: str):
    steps = [system_message(system_prompt)]
    if cot:
        steps.append(chain_of_thought())
    steps.append(generate())
    if critique:
        steps.append(self_critique())
    return steps


# --------------------------------------------------------------------------
# Adapter interface: every dataset type provides a loader + a scorer +
# a system prompt tailored to its answer format. Nothing else varies.
# --------------------------------------------------------------------------

@dataclass
class TaskAdapter:
    name: str
    system_prompt: str
    load: Callable[[], Dataset]
    scorer: Callable[[], Scorer]


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

MATH_SUBJECT = "algebra"  # swap for a harder subject if this is near-ceiling

MATH_SYSTEM = (
    "Solve the math problem. Show your reasoning, then give the final "
    "answer wrapped in \\boxed{...} on its own line at the end."
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


def _clean_latex(ans: str) -> str:
    """Normalize superficial LaTeX/formatting differences that don't change
    meaning: \\dfrac/\\tfrac -> \\frac, \\left \\right removed, spacing
    commands removed, a leading "x \\in " / "x = " variable prefix
    stripped, whitespace collapsed, trailing period and $ removed."""
    s = ans.strip()
    s = re.sub(r"\\d?frac", r"\\frac", s)          # \dfrac, \tfrac -> \frac
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;:!]", "", s)                  # LaTeX spacing commands
    s = re.sub(r"^[a-zA-Z]\s*(\\in|=)\s*", "", s)   # "x \in ", "x = " prefix
    s = s.replace("$", "").rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    return s


def _sympy_equivalent(a: str, b: str) -> bool | None:
    """Try to parse both answers as math expressions and check exact
    symbolic/numeric equivalence (catches \\frac{9}{7} == 9/7 == 1.2857...,
    \\frac{11}{2} == 5.5, etc). Returns None if sympy isn't installed or
    either side fails to parse -- caller should fall back to string match
    in that case, not treat None as "not equal"."""
    try:
        from sympy import sympify, simplify
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return None

    def _latex_to_sympy_str(s: str) -> str:
        """Manual \\frac{a}{b} -> (a)/(b), \\sqrt{x} -> sqrt(x) conversion,
        used when the optional antlr4 LaTeX parser isn't installed. Handles
        one level of nesting, which covers the vast majority of MATH
        dataset answers (e.g. \\frac{11}{2}, \\frac{1}{\\sqrt{2}})."""
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
            s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
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
        # last resort: manually rewrite common LaTeX macros, then sympify
        try:
            return sympify(_latex_to_sympy_str(s))
        except Exception:
            return None

    ea, eb = _parse(a), _parse(b)
    if ea is None or eb is None:
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
# NOTE: this adapter needs a Docker sandbox to actually execute code safely.
# It's included here to show the pattern; wire up sandbox="docker" on the
# Task before running it for real. Left unregistered by default until then
# -- see the note at the bottom of the registry.

def _load_humaneval() -> Dataset:
    def record_to_sample(record):
        return Sample(
            input=record["prompt"],
            target=record["test"],  # a pytest-style test string
            metadata={"entry_point": record["entry_point"]},
        )

    return hf_dataset(
        "openai_humaneval", split="test",
        sample_fields=record_to_sample, shuffle=False,
    )


HUMANEVAL_SYSTEM = (
    "Complete the Python function. Return ONLY the code, no explanation."
)


@scorer(metrics=[accuracy(), stderr()])
def code_execution_match():
    """Placeholder scorer shape -- real version executes state.output
    against target's test code inside the sandbox and checks pass/fail.
    Wire this up when you add sandbox="docker" to the Task below."""

    async def score(state: TaskState, target: Target):
        # TODO: run in sandbox, e.g. via inspect_ai.util.sandbox().exec(...)
        raise NotImplementedError(
            "code_execution_match needs sandbox=\"docker\" wired up on the "
            "Task before this adapter is usable -- see project step 7."
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
    # "humaneval": TaskAdapter("humaneval", HUMANEVAL_SYSTEM, _load_humaneval,
    #                           code_execution_match),  # needs sandbox first
}


@task
def elicit(suite: str = "math", cot: bool = False, critique: bool = False):
    if suite not in ADAPTERS:
        raise ValueError(
            f"Unknown suite '{suite}'. Available: {list(ADAPTERS)}. "
            f"To add one, register a new TaskAdapter above."
        )
    adapter = ADAPTERS[suite]
    return Task(
        dataset=adapter.load(),
        solver=build_solver(cot, critique, adapter.system_prompt),
        scorer=adapter.scorer(),
    )
