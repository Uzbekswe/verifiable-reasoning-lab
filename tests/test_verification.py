from reasonlab.verification import extract_final_candidate, verify_task


def numeric_task(answer="1/2"):
    return {"answer": answer, "answer_type": "numeric"}


def test_boxed_extraction_handles_nested_braces():
    result = extract_final_candidate(r"Work: simplify. Final: \boxed{\frac{1}{2}}")
    assert result.candidate == r"\frac{1}{2}"
    assert result.method == "boxed"


def test_numeric_verifier_accepts_equivalent_forms():
    result = verify_task(numeric_task(), r"Therefore \boxed{0.5}")
    assert result.status == "correct"
    assert result.reason == "symbolic_equivalence"


def test_logic_verifier_is_case_and_punctuation_tolerant():
    result = verify_task({"answer": "Cleo", "answer_type": "logic"}, r"Final answer: \boxed{cleo.}")
    assert result.status == "correct"


def test_missing_marker_is_a_parse_error():
    result = verify_task(numeric_task("4"), "The answer is probably four.")
    assert result.status == "parse_error"
    assert result.reason == "no_explicit_final_answer"


def test_unknown_answer_type_is_a_verifier_error():
    result = verify_task({"answer": "x", "answer_type": "unknown"}, r"\boxed{x}")
    assert result.status == "verifier_error"
