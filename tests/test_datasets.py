import json

from reasonlab.datasets import SPLIT_COUNTS, build_tasks, load_external_math500, write_manifests


def test_dataset_generation_is_reproducible_and_disjoint(tmp_path):
    first = build_tasks(123)
    second = build_tasks(123)
    assert {split: [task.to_dict() for task in rows] for split, rows in first.items()} == {
        split: [task.to_dict() for task in rows] for split, rows in second.items()
    }
    assert {split: len(rows) for split, rows in first.items()} == SPLIT_COUNTS
    ids = [task.task_id for rows in first.values() for task in rows]
    assert len(ids) == len(set(ids))
    metadata = write_manifests(tmp_path, seed=123)
    assert metadata["counts"] == SPLIT_COUNTS
    assert all(info["sha256"] for info in metadata["files"].values())


def test_external_math500_loader_does_not_bundle_or_mutate_source(tmp_path):
    source = tmp_path / "math500.json"
    source.write_text(json.dumps([{"problem": "1+1", "answer": "2", "subject": "algebra", "level": 1}]))
    rows = load_external_math500(source)
    assert rows[0]["task_id"] == "math500-0000"
    assert rows[0]["split"] == "external"
    assert rows[0]["license"] == "user-supplied-verify-before-use"
