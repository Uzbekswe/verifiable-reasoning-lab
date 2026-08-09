"""Model-agnostic baseline evaluation over frozen task manifests."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .datasets import render_prompt
from .verification import verify_task


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    correct: int
    incorrect: int
    parse_errors: int
    verifier_errors: int
    accuracy: float
    parse_rate: float
    verifier_error_rate: float
    mean_generated_token_count: float
    mean_latency_seconds: float
    by_family: dict
    by_difficulty: dict

    def to_dict(self):
        return asdict(self)


def _generation_fields(result):
    if hasattr(result, "text"):
        return result.text, {
            "generated_token_count": result.generated_token_count,
            "latency_seconds": result.elapsed_seconds,
            "tokens_per_second": result.tokens_per_second,
            "device": result.device,
        }
    return str(result), {}


def evaluate_tasks(tasks: list[dict], generate: Callable[[str], object], output_path=None) -> dict:
    records = []
    counts = {"correct": 0, "incorrect": 0, "parse_error": 0, "verifier_error": 0}
    started = time.perf_counter()
    for task in tasks:
        prompt = render_prompt(task)
        try:
            generation = generate(prompt)
            raw_output, generation_fields = _generation_fields(generation)
            checked = verify_task(task, raw_output)
            counts[checked.status] += 1
            records.append({
                "task_id": task["task_id"],
                "split": task["split"],
                "family": task["family"],
                "difficulty": task["difficulty"],
                "prompt": prompt,
                "raw_output": raw_output,
                "verification": checked.to_dict(),
                **generation_fields,
            })
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            counts["verifier_error"] += 1
            records.append({
                "task_id": task["task_id"],
                "split": task["split"],
                "family": task["family"],
                "difficulty": task["difficulty"],
                "prompt": prompt,
                "raw_output": None,
                "verification": {
                    "status": "verifier_error",
                    "candidate": None,
                    "expected": str(task["answer"]),
                    "extraction_method": None,
                    "reason": f"generation_exception:{type(exc).__name__}:{exc}",
                },
            })
    total = len(tasks)
    parsed = counts["correct"] + counts["incorrect"]
    token_counts = [record["generated_token_count"] for record in records if "generated_token_count" in record]
    latencies = [record["latency_seconds"] for record in records if "latency_seconds" in record]

    def grouped(field: str) -> dict:
        groups = {}
        for record in records:
            key = record[field]
            group = groups.setdefault(key, {"total": 0, "correct": 0, "parse_errors": 0, "verifier_errors": 0})
            group["total"] += 1
            status = record["verification"]["status"]
            if status == "correct":
                group["correct"] += 1
            elif status == "parse_error":
                group["parse_errors"] += 1
            elif status == "verifier_error":
                group["verifier_errors"] += 1
            group["accuracy"] = group["correct"] / group["total"]
        return groups

    summary = EvaluationSummary(
        total=total,
        correct=counts["correct"],
        incorrect=counts["incorrect"],
        parse_errors=counts["parse_error"],
        verifier_errors=counts["verifier_error"],
        accuracy=counts["correct"] / total if total else 0.0,
        parse_rate=parsed / total if total else 0.0,
        verifier_error_rate=counts["verifier_error"] / total if total else 0.0,
        mean_generated_token_count=sum(token_counts) / len(token_counts) if token_counts else 0.0,
        mean_latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
        by_family=grouped("family"),
        by_difficulty=grouped("difficulty"),
    )
    result = {"summary": summary.to_dict(), "elapsed_seconds": time.perf_counter() - started, "records": records}
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
