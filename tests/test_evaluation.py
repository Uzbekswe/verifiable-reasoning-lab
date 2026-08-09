from reasonlab.evaluation import evaluate_tasks


def test_evaluation_keeps_correct_and_parse_failures_separate():
    tasks = [
        {"task_id": "a", "split": "test", "family": "arithmetic", "difficulty": "easy", "prompt": "2+2", "answer": "4", "answer_type": "numeric"},
        {"task_id": "b", "split": "test", "family": "arithmetic", "difficulty": "easy", "prompt": "3+3", "answer": "6", "answer_type": "numeric"},
    ]
    outputs = iter([r"\boxed{4}", "I forgot the required marker."])
    result = evaluate_tasks(tasks, lambda _: next(outputs))
    assert result["summary"]["correct"] == 1
    assert result["summary"]["parse_errors"] == 1
    assert result["summary"]["accuracy"] == 0.5
