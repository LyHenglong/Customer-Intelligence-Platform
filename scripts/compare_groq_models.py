"""Compare Groq models on this project's real agent prompts.

Runs the three AI Agent Layer prompts (churn explanation, retention
outreach, retrain summary) with the exact system prompts, prompt builders
and generation settings the agents use in production, once per model and
repetition, and reports per model and task:

- latency (median and p90 wall-clock seconds per call)
- tokens (prompt, completion) and estimated cost
- failures (API errors, empty completions after retries)
- grounding: numbers in the output that don't appear in the prompt, via the
  same number extraction the AI assistant's guardrail uses
- task checks: the explanation states the real probability, the outreach
  names the recommended service, the summary cites the real AUC values

Inputs are the real examples documented in docs/ai-layer.md (customer
CUST0000269643 and the 20260910T154013Z retrain), so results are comparable
with the outputs recorded there.

Needs GROQ_API_KEY and network access to api.groq.com. Every call costs a
fraction of a cent; the defaults make 3 tasks x 2 models x 5 reps = 30 calls.

Usage:
    python -m scripts.compare_groq_models
    python -m scripts.compare_groq_models --models openai/gpt-oss-20b openai/gpt-oss-120b --reps 10
    python -m scripts.compare_groq_models --out report/groq_model_comparison.md
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.agents import explanation_agent, outreach_agent, retrain_summary_agent
from src.agents.groq_client import AgentCallFailed, complete
from src.ai.guardrails.grounding import extract_numbers
from src.ai.observability.cost import estimate_cost

DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

CHURN_PROBABILITY = 0.326
SHAP_DETAILS = [
    {"feature": "contract", "shap_value": 0.544, "direction": "increases risk"},
    {"feature": "num_complaints", "shap_value": 0.380, "direction": "increases risk"},
    {"feature": "num_service_calls", "shap_value": 0.369, "direction": "increases risk"},
    {"feature": "late_payments", "shap_value": 0.286, "direction": "increases risk"},
    {"feature": "age", "shap_value": 0.265, "direction": "increases risk"},
]
EXPLANATION_FOR_OUTREACH = (
    "The model estimates a 32.6% chance that the customer will churn, driven primarily by the fact that they are "
    "on a contract, have logged multiple complaints, and have made several service calls, all of which raise the "
    "risk. Additional contributors are late payments and a younger age, which also increase the likelihood of churn."
)
RECOMMENDED_SERVICE = "Internet Service"
CURRENT_METRICS = {
    "version": "20260910T154013Z",
    "precision_churn": 0.1534,
    "recall_churn": 0.6003,
    "f1_churn": 0.2411,
    "roc_auc": 0.6564,
}
PREVIOUS_METRICS = {
    "version": "20260910T124230Z",
    "precision_churn": 0.1614,
    "recall_churn": 0.6000,
    "f1_churn": 0.2528,
    "roc_auc": 0.6693,
}
DRIFT_SUMMARY = {"drift_detected": False, "max_psi": 0.0003}


@dataclass
class Task:
    name: str
    system_prompt: str
    user_prompt: str
    max_tokens: int
    temperature: float
    check_name: str
    check: callable


def _tasks() -> list[Task]:
    return [
        Task(
            name="explanation",
            system_prompt=explanation_agent.SYSTEM_PROMPT,
            user_prompt=explanation_agent.build_prompt(CHURN_PROBABILITY, SHAP_DETAILS),
            max_tokens=400,
            temperature=0.3,
            check_name="states 32.6%",
            check=lambda text: "32.6" in text,
        ),
        Task(
            name="outreach",
            system_prompt=outreach_agent.SYSTEM_PROMPT,
            user_prompt=outreach_agent.build_prompt(EXPLANATION_FOR_OUTREACH, RECOMMENDED_SERVICE),
            max_tokens=500,
            temperature=0.4,
            check_name="names the service",
            check=lambda text: outreach_agent.mentions_service(text, RECOMMENDED_SERVICE),
        ),
        Task(
            name="retrain_summary",
            system_prompt=retrain_summary_agent.SYSTEM_PROMPT,
            user_prompt=retrain_summary_agent.build_prompt(CURRENT_METRICS, PREVIOUS_METRICS, DRIFT_SUMMARY),
            max_tokens=400,
            temperature=0.3,
            check_name="cites both AUCs",
            check=lambda text: "0.6564" in text and "0.6693" in text,
        ),
    ]


@dataclass
class Result:
    latencies: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    failures: int = 0
    check_passes: int = 0
    ungrounded_outputs: int = 0
    ungrounded_examples: list[str] = field(default_factory=list)
    sample: str = ""


def _ungrounded_numbers(output: str, prompt: str) -> list[str]:
    return sorted(extract_numbers(output) - extract_numbers(prompt))


def run(models: list[str], reps: int) -> dict[tuple[str, str], Result]:
    results: dict[tuple[str, str], Result] = {}
    tasks = _tasks()
    # Interleave models within each repetition so both see the same
    # moment-to-moment API load, rather than one model's calls all landing
    # in a slower window.
    for rep in range(reps):
        for task in tasks:
            for model in models:
                r = results.setdefault((model, task.name), Result())
                start = time.perf_counter()
                try:
                    response = complete(
                        task.system_prompt,
                        task.user_prompt,
                        model=model,
                        max_tokens=task.max_tokens,
                        temperature=task.temperature,
                    )
                except AgentCallFailed as exc:
                    r.failures += 1
                    print(f"  rep {rep + 1} {task.name:<16} {model:<22} FAILED: {exc}")
                    continue
                elapsed = time.perf_counter() - start
                r.latencies.append(elapsed)
                r.prompt_tokens.append(response.prompt_tokens)
                r.completion_tokens.append(response.completion_tokens)
                r.check_passes += task.check(response.text)
                ungrounded = _ungrounded_numbers(response.text, task.user_prompt)
                if ungrounded:
                    r.ungrounded_outputs += 1
                    r.ungrounded_examples.extend(ungrounded)
                if not r.sample:
                    r.sample = response.text
                print(
                    f"  rep {rep + 1} {task.name:<16} {model:<22} {elapsed:5.2f}s {response.completion_tokens:>4} tok"
                )
    return results


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def report(results: dict[tuple[str, str], Result], models: list[str], reps: int) -> str:
    tasks = _tasks()
    checks = {t.name: t.check_name for t in tasks}
    lines = [
        "# Groq model comparison",
        "",
        f"Run {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} with `scripts/compare_groq_models.py`: "
        f"{reps} repetitions of each AI Agent Layer prompt per model, production prompts and settings.",
        "",
        "| Task | Model | Median s | p90 s | Completion tok (mean) | Cost / 1k calls | Failures | Task check | "
        "Outputs with ungrounded numbers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in tasks:
        for model in models:
            r = results.get((model, task.name), Result())
            ok = len(r.latencies)
            if ok:
                cost = estimate_cost(model, statistics.mean(r.prompt_tokens), statistics.mean(r.completion_tokens))
                lines.append(
                    f"| {task.name} | `{model}` | {statistics.median(r.latencies):.2f} | {_p90(r.latencies):.2f} | "
                    f"{statistics.mean(r.completion_tokens):.0f} | ${cost * 1000:.3f} | {r.failures}/{reps} | "
                    f"{checks[task.name]}: {r.check_passes}/{ok} | {r.ungrounded_outputs}/{ok} |"
                )
            else:
                lines.append(f"| {task.name} | `{model}` | - | - | - | - | {r.failures}/{reps} | - | - |")

    lines += [
        "",
        "Ungrounded numbers are numbers in the output that never appear in the prompt. The check is exact-text, so a "
        "correct derived figure (a percentage written from a fraction, a difference between two metrics) also counts; "
        "read the examples below before treating it as hallucination.",
        "",
    ]
    for task in tasks:
        for model in models:
            r = results.get((model, task.name))
            if r and r.ungrounded_examples:
                lines.append(f"- {task.name}, `{model}`: {', '.join(sorted(set(r.ungrounded_examples)))}")
    lines += ["", "## Sample outputs (first successful call)", ""]
    for task in tasks:
        for model in models:
            r = results.get((model, task.name))
            if r and r.sample:
                quoted = "\n".join("> " + line for line in r.sample.splitlines())
                lines += [f"**{task.name}, `{model}`**", "", quoted, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", help="also write the markdown report to this path")
    args = parser.parse_args()

    results = run(args.models, args.reps)
    text = report(results, args.models, args.reps)
    print()
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
