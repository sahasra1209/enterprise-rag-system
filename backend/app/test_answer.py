
from unittest.mock import patch
from answer import generate_answer


def run_test(question, expected_text, test_name):
    print(f"\nTest: {test_name}")

    with patch("builtins.print"):
        result = generate_answer(question)

    # Check the returned answer instead of terminal output.
    answer = result.get("answer", "").lower()

    if expected_text.lower() in answer:
        print("PASS")
        return True

    print("FAIL")
    print("Expected:", expected_text)
    print("Actual:", answer)
    print("Status:", result.get("overall_status"))
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
