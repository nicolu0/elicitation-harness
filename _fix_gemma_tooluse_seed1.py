import json
from pathlib import Path

from inspect_ai import eval as inspect_eval

from elicit_task import elicit
from run_shapley_sweep import MAX_TOKENS, EXTRA_BODY, get_accuracy

model = "openai-api/deepinfra/google/gemma-3-27b-it"
summary_path = Path("results/shapley_sweep_math_openai-api-deepinfra-google-gemma-3-27b-it.json")

with open(summary_path) as f:
    data = json.load(f)

log = inspect_eval(
    elicit(suite="math", tool_use=True),
    model=model,
    limit=500,
    temperature=0.7,
    seed=1,
    log_dir="./logs",
    max_connections=20,
    max_tokens=MAX_TOKENS,
    extra_body=EXTRA_BODY,
)[0]
acc = get_accuracy(log)
log_path = str(log.location) if hasattr(log, "location") else None
print(f"seed=1 accuracy={acc} status={log.status} log={log_path}")

if acc == acc:  # not nan
    for r in data["results"]["tool_use"]:
        if r["seed"] == 1:
            r["accuracy"] = acc
            r["log"] = log_path
            r["limit"] = 500
            break
    with open(summary_path, "w") as f:
        json.dump(data, f, indent=2)
    print("PATCHED summary file successfully")
else:
    print("STILL FAILED -- not patching, needs investigation")
