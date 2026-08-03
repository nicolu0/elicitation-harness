"""
Parametrized elicitation task.

Holds the MODEL fixed and toggles scaffold COMPONENTS on/off, so we can
measure how much each component contributes to *measured* capability.

Start here: GSM8K (numeric, verifiable answers) with two no-sandbox
components — chain-of-thought and self-critique — so this runs with just an
API key. Tool use and SWE-bench come next and need a Docker sandbox.

Run a single config directly:
    inspect eval elicit_task.py --model openai/gpt-4o-mini --limit 20 -T cot=true
Or sweep all configs with run_ablation.py.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    chain_of_thought,
    generate,
    self_critique,
    system_message,
)

SYSTEM = (
    "Solve the math problem. Show brief reasoning, then write the final "
    "answer as a plain number on its own line at the end."
)


def record_to_sample(record):
    # GSM8K targets look like "...\n#### 1080"
    target = record["answer"].split("####")[-1].strip().replace(",", "")
    return Sample(input=record["question"], target=target)


def build_solver(cot: bool, critique: bool):
    """Assemble the solver chain from component toggles."""
    steps = [system_message(SYSTEM)]
    if cot:
        steps.append(chain_of_thought())
    steps.append(generate())
    if critique:
        steps.append(self_critique())
    return steps


@task
def elicit(cot: bool = False, critique: bool = False):
    return Task(
        dataset=hf_dataset(
            "openai/gsm8k",
            name="main",
            split="test",
            sample_fields=record_to_sample,
            # IMPORTANT: keep ordering fixed so EVERY config is scored on the
            # exact same samples. If configs saw different tasks, the ablation
            # would be invalid.
            shuffle=False,
        ),
        solver=build_solver(cot, critique),
        scorer=match(location="end", numeric=True),
    )
