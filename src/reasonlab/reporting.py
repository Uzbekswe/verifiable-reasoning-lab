"""Artifact-backed portfolio summaries and dependency-free SVG figures."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

REPORT_RUNS = (
    "ch03_baseline.json",
    "ch04_sample.json",
    "ch05_self_refinement.json",
    "m5_fixed_budget_test.json",
    "m5_adaptive_test.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"artifact must contain a JSON object: {path}")
    return value


def _run_summary(name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact.get("summary")
    provenance = artifact.get("provenance", {})
    if not isinstance(summary, dict) or "accuracy" not in summary:
        raise ValueError(f"artifact lacks an evaluation summary: {name}")
    records = artifact.get("records", [])
    policy_config = artifact.get("policy_config", {})
    split = records[0].get("split") if records else None
    return {
        "artifact": name,
        "policy": policy_config.get("policy", "base"),
        "split": split,
        "total": summary["total"],
        "correct": summary["correct"],
        "incorrect": summary["incorrect"],
        "parse_errors": summary["parse_errors"],
        "verifier_errors": summary["verifier_errors"],
        "accuracy": summary["accuracy"],
        "mean_generated_token_count": summary["mean_generated_token_count"],
        "mean_latency_seconds": summary["mean_latency_seconds"],
        "by_family": summary.get("by_family", {}),
        "by_difficulty": summary.get("by_difficulty", {}),
        "model_sha256": provenance.get("model_sha256"),
        "device": provenance.get("device"),
    }


def build_portfolio_summary(
    run_dir: str | Path = "artifacts/runs",
    manifest_path: str | Path = "data/manifests/manifest.json",
) -> dict[str, Any]:
    """Validate the final evidence set and return a deterministic summary."""
    run_dir = Path(run_dir)
    runs = [_run_summary(name, _load_json(run_dir / name)) for name in REPORT_RUNS]
    model_hashes = {run["model_sha256"] for run in runs if run["model_sha256"]}
    if len(model_hashes) != 1:
        raise ValueError(f"evaluation artifacts disagree on model hash: {sorted(model_hashes)}")
    fixed = next(run for run in runs if run["artifact"] == "m5_fixed_budget_test.json")
    adaptive = next(run for run in runs if run["artifact"] == "m5_adaptive_test.json")
    fixed_tokens = fixed["mean_generated_token_count"]
    adaptive_tokens = adaptive["mean_generated_token_count"]
    fixed_latency = fixed["mean_latency_seconds"]
    adaptive_latency = adaptive["mean_latency_seconds"]
    return {
        "project": "verifiable-reasoning-lab",
        "evidence_scope": "original verifier-backed math and logic suite",
        "model_sha256": next(iter(model_hashes)),
        "runs": runs,
        "adaptive_comparison": {
            "fixed_accuracy": fixed["accuracy"],
            "adaptive_accuracy": adaptive["accuracy"],
            "accuracy_delta_percentage_points": 100 * (adaptive["accuracy"] - fixed["accuracy"]),
            "token_savings_fraction": 1 - adaptive_tokens / fixed_tokens,
            "latency_delta_seconds": adaptive_latency - fixed_latency,
        },
        "manifest": _load_json(Path(manifest_path)),
        "claims": {
            "base_model": "Qwen3-0.6B from rasbt/qwen3-from-scratch",
            "full_model_grpo_claim": False,
            "full_distillation_claim": False,
            "cloud_spend": 0.0,
        },
    }


def _svg_text(x: float, y: float, value: str, size: int = 13, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}">{html.escape(value)}</text>'


def write_accuracy_compute_svg(summary: dict[str, Any], path: str | Path) -> None:
    points = [
        (run["policy"], run["mean_generated_token_count"], run["accuracy"] * 100)
        for run in summary["runs"]
    ]
    width, height = 900, 500
    left, bottom, top, right = 80, 420, 60, 40
    max_tokens = max(point[1] for point in points) * 1.15
    max_accuracy = max(40.0, max(point[2] for point in points) * 1.2)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, "Accuracy versus generated-token budget", 18, "middle"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="black"/>',
        _svg_text(width / 2, height - 12, "mean generated tokens per task", 14, "middle"),
        _svg_text(18, height / 2, "accuracy (%)", 14, "middle"),
    ]
    for tick in range(0, 41, 10):
        y = bottom - (tick / max_accuracy) * (bottom - top)
        body.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        body.append(_svg_text(left - 10, y + 4, str(tick), 11, "end"))
    for label, tokens, accuracy in points:
        x = left + (tokens / max_tokens) * (width - left - right)
        y = bottom - (accuracy / max_accuracy) * (bottom - top)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#2563eb"/>')
        body.append(_svg_text(x + 9, y + 4, label, 11))
    body.append("</svg>")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body) + "\n", encoding="utf-8")


def write_failure_slices_svg(summary: dict[str, Any], path: str | Path) -> None:
    adaptive = next(run for run in summary["runs"] if run["artifact"] == "m5_adaptive_test.json")
    families = list(adaptive["by_family"])
    width, height = 900, 420
    left, top, row_height, bar_width = 170, 70, 70, 600
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 30, "Adaptive test-set failure slices", 18, "middle"),
        _svg_text(width - 150, height - 15, "task count", 13, "middle"),
    ]
    for index, family in enumerate(families):
        row = adaptive["by_family"][family]
        y = top + index * row_height
        body.append(_svg_text(left - 12, y + 22, family, 13, "end"))
        correct_w = bar_width * row["correct"] / row["total"]
        parse_w = bar_width * row["parse_errors"] / row["total"]
        body.append(f'<rect x="{left}" y="{y}" width="{correct_w:.1f}" height="28" fill="#16a34a"/>')
        body.append(f'<rect x="{left + correct_w:.1f}" y="{y}" width="{parse_w:.1f}" height="28" fill="#f97316"/>')
        body.append(_svg_text(left + bar_width + 10, y + 20, f'{row["correct"]} correct / {row["parse_errors"]} parse', 11))
    body.extend([
        '<rect x="170" y="355" width="14" height="14" fill="#16a34a"/>',
        _svg_text(190, 367, "correct", 12),
        '<rect x="270" y="355" width="14" height="14" fill="#f97316"/>',
        _svg_text(290, 367, "parse error", 12),
        "</svg>",
    ])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body) + "\n", encoding="utf-8")
