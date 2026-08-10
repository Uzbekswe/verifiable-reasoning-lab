import json

from reasonlab.reporting import (
    build_portfolio_summary,
    write_accuracy_compute_svg,
    write_failure_slices_svg,
)


def _artifact(policy, split="test"):
    return {
        "summary": {
            "total": 2,
            "correct": 1,
            "incorrect": 0,
            "parse_errors": 1,
            "verifier_errors": 0,
            "accuracy": 0.5,
            "mean_generated_token_count": 10.0,
            "mean_latency_seconds": 1.0,
            "by_family": {"arithmetic": {"total": 2, "correct": 1, "parse_errors": 1}},
            "by_difficulty": {},
        },
        "records": [{"split": split}],
        "policy_config": {"policy": policy},
        "provenance": {"model_sha256": "same", "device": "cpu"},
    }


def test_reporting_validates_hashes_and_writes_figures(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for name, policy in (
        ("ch03_baseline.json", "base"),
        ("ch04_sample.json", "sample"),
        ("ch05_self_refinement.json", "self_refinement"),
        ("m5_fixed_budget_test.json", "fixed_verifier_budget"),
        ("m5_adaptive_test.json", "adaptive_budget"),
    ):
        (run_dir / name).write_text(json.dumps(_artifact(policy)), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed": 1}), encoding="utf-8")
    summary = build_portfolio_summary(run_dir, manifest)
    assert summary["model_sha256"] == "same"
    assert summary["adaptive_comparison"]["token_savings_fraction"] == 0.0
    write_accuracy_compute_svg(summary, tmp_path / "accuracy.svg")
    write_failure_slices_svg(summary, tmp_path / "failures.svg")
    assert (tmp_path / "accuracy.svg").read_text().startswith("<svg")
    assert (tmp_path / "failures.svg").read_text().startswith("<svg")
