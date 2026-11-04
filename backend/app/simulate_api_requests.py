
import json
import time
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/ask"

# Start with 3 real API requests.
TOTAL_REQUESTS = 50

questions = [
    "What was 3M capital expenditure (CapEx) in fiscal year 2018?",
    "What was 3M's revenue in 2018?",
    "How many employees did 3M have in 2018?"
]


def send_query(question, number):
    payload = json.dumps({
        "question": question
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start_time = time.time()

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        elapsed = round(time.time() - start_time, 2)

        print(f"\nRequest {number}/{TOTAL_REQUESTS}")
        print("Question:", question)
        print("Status:", result.get("overall_status"))
        print("Answer:", result.get("answer", "")[:150])
        print("Elapsed time:", elapsed, "seconds")

    except urllib.error.HTTPError as error:
        print(f"\nRequest {number} failed with HTTP error:", error.code)
        print(error.read().decode("utf-8", errors="replace"))

    except Exception as error:
        print(f"\nRequest {number} failed:", error)


def main():
    print("Starting live RAG API simulation...")

    for number in range(1, TOTAL_REQUESTS + 1):
        question = questions[(number - 1) % len(questions)]
        send_query(question, number)

    print("\nLive API simulation finished.")


if __name__ == "__main__":
    main()
