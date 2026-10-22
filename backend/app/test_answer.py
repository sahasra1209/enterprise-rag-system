
from unittest.mock import patch
from answer import generate_answer


def run_test(question, expected_text, test_name):
    print(f"\nTest: {test_name}")

    with patch("builtins.print") as mock_print:
        generate_answer(question)

    output = " ".join(
        str(call) for call in mock_print.call_args_list
    ).lower()

    if expected_text.lower() in output:
        print("PASS")
        return True

    print("FAIL")
    print("Expected:", expected_text)
    print("Check the answer-generation output.")
    return False


def main():
    tests = [
        (
            "Who approves leave requests?",
            "managers must approve leave requests",
            "Leave approval"
        ),
        (
            "Where can employees request leave?",
            "hr portal",
            "Leave request location"
        ),
        (
            "What is the company's maternity leave policy?",
            "couldn't find a supported answer",
            "Unsupported question refusal"
        ),
    ]

    passed = 0

    for question, expected, name in tests:
        if run_test(question, expected, name):
            passed += 1

    print(f"\nAnswer tests: {passed}/{len(tests)} passed")

    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
