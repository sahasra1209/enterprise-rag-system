
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from answer import generate_answer


FAKE_RESULTS = {
    "documents": [[
        "Managers must approve leave requests."
    ]],
    "metadatas": [[{
        "source": "test policy.pdf",
        "page_number": 1,
        "chunk_number": 2
    }]],
    "distances": [[0.5]]
}


def run_test(name, model_answer, expected_output):
    captured = StringIO()

    with (
        patch("answer.search_documents", return_value=FAKE_RESULTS),
        patch(
            "answer.ollama.chat",
            return_value={
                "message": {"content": model_answer}
            }
        ),
        redirect_stdout(captured)
    ):
        generate_answer("Who approves leave requests?")

    output = captured.getvalue().lower()

    if expected_output.lower() in output:
        print(f"PASS: {name}")
        return True

    print(f"FAIL: {name}")
    print("Actual output:")
    print(captured.getvalue())
    return False


def main():
    tests = [
        (
            "Valid citation",
            "Managers must approve leave requests. [E1]",
            "Managers must approve leave requests."
        ),
        (
            "Invalid citation rejected",
            "Managers approve leave. [E9]",
            "invalid evidence references"
        ),
        (
            "Missing citation rejected",
            "Managers must approve leave requests.",
            "verifiable evidence citation"
        ),
        (
            "Unsupported answer handled",
            "I couldn't find a supported answer in the provided documents.",
            "couldn't find a supported answer"
        ),
    ]

    passed = 0

    for name, model_answer, expected in tests:
        if run_test(name, model_answer, expected):
            passed += 1

    print(f"\nCitation validation tests: {passed}/{len(tests)} passed")

    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
