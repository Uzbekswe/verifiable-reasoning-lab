"""Deterministic original task generation and frozen split manifests."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

GENERATOR_VERSION = "chapter3-original-v1"
DEFAULT_SEED = 20260809
SPLIT_COUNTS = {"train": 400, "validation": 80, "test": 120}
FAMILIES = ("arithmetic", "algebra", "sequence", "logic")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    difficulty: str
    prompt: str
    answer: str
    answer_type: str
    prompt_version: str = "chapter3-prompt-v1"
    verifier_version: str = "chapter3-v1"
    source: str = "project-original-generated"
    license: str = "project-original"
    split: str = ""

    def to_dict(self):
        return asdict(self)


def _task(split: str, family: str, index: int, difficulty: str, prompt: str, answer: str, answer_type: str):
    return Task(
        task_id=f"{split}-{family}-{index:04d}",
        family=family,
        difficulty=difficulty,
        prompt=prompt,
        answer=str(answer),
        answer_type=answer_type,
        split=split,
    )


def _make_family(split: str, family: str, count: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        difficulty = "easy" if index % 3 else "medium"
        if family == "arithmetic":
            variant = index % 4
            a, b = rng.randint(2, 99), rng.randint(2, 49)
            if variant == 0:
                prompt, answer = f"Compute {a} + {b}.", a + b
            elif variant == 1:
                prompt, answer = f"Compute {a + b} - {b}.", a
            elif variant == 2:
                prompt, answer = f"Compute {a} × {b}.", a * b
            else:
                product = a * b
                prompt, answer = f"Compute {product} / {b}.", a
            tasks.append(_task(split, family, index, difficulty, prompt, str(answer), "numeric"))
        elif family == "algebra":
            x = rng.randint(-20, 40)
            coefficient = rng.choice((2, 3, 4, 5))
            offset = rng.randint(-15, 15)
            result = coefficient * x + offset
            prompt = f"Solve for x: {coefficient}x + ({offset}) = {result}."
            tasks.append(_task(split, family, index, difficulty, prompt, str(x), "numeric"))
        elif family == "sequence":
            start = rng.randint(-20, 20)
            step = rng.randint(2, 12) * (-1 if index % 5 == 0 else 1)
            values = [start + step * n for n in range(4)]
            prompt = f"Sequence {index + 1}: find the next number: {', '.join(map(str, values))}, __."
            tasks.append(_task(split, family, index, difficulty, prompt, str(values[-1] + step), "numeric"))
        else:
            names = rng.sample(["Ava", "Ben", "Cleo", "Dina", "Eli", "Faye"], 3)
            heights = sorted(rng.sample(range(140, 201), 3), reverse=True)
            tallest, middle, shortest = names
            prompt = (
                f"{tallest} is {heights[0]} cm tall, {middle} is {heights[1]} cm tall, "
                f"and {shortest} is {heights[2]} cm tall. Who is the shortest?"
            )
            tasks.append(_task(split, family, index, difficulty, prompt, shortest, "logic"))
    return tasks


def build_tasks(seed: int = DEFAULT_SEED) -> dict[str, list[Task]]:
    splits: dict[str, list[Task]] = {}
    for split, total in SPLIT_COUNTS.items():
        per_family, remainder = divmod(total, len(FAMILIES))
        rows = []
        for family_index, family in enumerate(FAMILIES):
            count = per_family + (family_index < remainder)
            rows.extend(_make_family(split, family, count, seed + family_index * 1000 + len(split)))
        splits[split] = rows
    return splits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifests(out_dir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = build_tasks(seed)
    files = {}
    for split, tasks in splits.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        files[split] = {"path": path.name, "count": len(tasks), "sha256": _sha256(path)}
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "counts": SPLIT_COUNTS,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_split(manifest_dir: str | Path, split: str) -> list[dict]:
    path = Path(manifest_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run `reasonlab data build` first")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_external_math500(path: str | Path) -> list[dict]:
    """Normalize a user-provided licensed MATH-500 JSON file without bundling it."""
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError("MATH-500 input must be a JSON list")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "problem" not in row or "answer" not in row:
            raise ValueError(f"MATH-500 row {index} must contain problem and answer")
        normalized.append({
            "task_id": f"math500-{index:04d}",
            "family": row.get("subject", "math500"),
            "difficulty": row.get("level", "unknown"),
            "prompt": row["problem"],
            "answer": str(row["answer"]),
            "answer_type": "numeric",
            "prompt_version": "external-math500",
            "verifier_version": "chapter3-v1",
            "source": str(source_path),
            "license": "user-supplied-verify-before-use",
            "split": "external",
        })
    return normalized


def render_prompt(task: dict) -> str:
    return (
        "You are a careful math and logic solver. Show brief work, then put only "
        "your final answer on a separate line using exactly \\boxed{ANSWER}.\n\n"
        f"Question: {task['prompt']}\n\nAnswer:"
    )
