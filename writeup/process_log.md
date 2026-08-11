# Process Log

Running record of what was actually tried, what worked, what didn't, and
why decisions were made. Written close to when things happened. This is
the raw material for the polished writeup later — don't clean it up
retroactively, just keep appending.

---

## Phase 1 — Infrastructure setup

Set up the repo on top of Inspect AI (UK AISI's eval framework) rather
than building model clients, dataset loaders, and logging from scratch.
Decision rationale: the novel contribution is the measurement idea
(component attribution, budget substitution, cross-model transfer), not
plumbing that already exists and is used by serious evaluators (METR,
AISI). Building the plumbing from scratch would have signaled the
opposite of what a senior engineer/researcher does — reinventing solved
infrastructure instead of building the novel layer on top of it.

- `uv venv --python 3.12`, installed `inspect-ai` + `inspect_evals`
- GitHub repo created public, MIT license, `.gitignore` for `.env` and
  `.venv/`
- Directory skeleton scaffolded: `src/elicit/{models,components,budget,
  tasks,runner,attribution,analysis}`, `experiments/`, `results/`,
  `figures/`, `suites/`, `writeup/`
- Smoke test: `inspect eval inspect_evals/gsm8k --model openai/gpt-4o-mini
  --limit 20` → accuracy 0.85-0.90 range, confirmed the full stack
  (model interface, dataset loading, scoring, logging) works end to end

