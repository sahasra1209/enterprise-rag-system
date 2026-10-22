
from search import search_documents


TEST_CASES = [
    {
        "question": "Who approves leave requests?",
        "expected_evidence": "managers must approve leave requests"
    },
    {
        "question": "Where can employees request leave?",
        "expected_evidence": "hr portal"
    },
]


def run_tests():
    passed = 0
    failed = 0

    for test in TEST_CASES:
        question = test["question"]
        expected = test["expected_evidence"].lower()

        print(f"\nTesting: {question}")

        try:
            results = search_documents(question, top_k=3)

            documents = results.get("documents", [[]])[0]
            retrieved_text = " ".join(documents).lower()

            if expected in retrieved_text:
                print("PASS: Expected evidence was retrieved.")
                passed += 1
            else:
                print("FAIL: Expected evidence was not retrieved.")
                failed += 1

        except Exception as error:
            print(f"FAIL: {error}")
            failed += 1

    print("\n========== TEST SUMMARY ==========")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
