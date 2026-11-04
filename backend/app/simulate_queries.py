
import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).resolve().parent / "analytics.db"

rng = random.Random(42)

questions = [
    ("Finance", "What was the company's total revenue in 2018?"),
    ("Finance", "What was the capital expenditure in 2018?"),
    ("Finance", "How much debt did the company report?"),
    ("Finance", "What were the operating expenses?"),
    ("Operations", "What were the major products manufactured?"),
    ("Operations", "How did production change during the year?"),
    ("Operations", "What were the manufacturing costs?"),
    ("Human Resources", "How many employees did the company have?"),
    ("Human Resources", "What changes occurred in the workforce?"),
    ("General", "Summarize the annual report."),
]

statuses = ["supported", "needs_review", "insufficient_evidence"]
status_weights = [0.70, 0.20, 0.10]

confidence_values = {
    "supported": 1.0,
    "needs_review": 0.5,
    "insufficient_evidence": 0.0
}


def simulate_queries():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # Remove previous simulation records if this script is run again.
    cursor.execute("""
        DELETE FROM query_logs
        WHERE substr(question, 1, 11) = '[SIMULATED]'
    """)

    records = []

    # Generate 60 records spread across the last seven days.
    for i in range(60):
        category, question = rng.choice(questions)
        status = rng.choices(
            statuses,
            weights=status_weights,
            k=1
        )[0]

        confidence = confidence_values[status]
        response_time = round(rng.uniform(1000, 45000), 2)
        token_count = rng.randint(20, 250)
        simulated_cost = (token_count / 1000) * 0.002

        days_ago = rng.randint(0, 6)
        hour = rng.randint(9, 18)
        minute = rng.randint(0, 59)

        query_datetime = (
            datetime.combine(
                datetime.now().date() - timedelta(days=days_ago),
                datetime.min.time()
            )
            + timedelta(hours=hour, minutes=minute)
        )

        records.append((
            query_datetime.isoformat(timespec="seconds"),
            f"[SIMULATED] {question}",
            "[]",
            f"Simulated response for {category} question.",
            status,
            confidence,
            response_time,
            token_count,
            simulated_cost
        ))

    cursor.executemany("""
        INSERT INTO query_logs (
            created_at,
            question,
            retrieved_chunks,
            answer,
            status,
            confidence_score,
            response_time_ms,
            estimated_token_count,
            simulated_cost_usd
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    connection.commit()

    print("Successfully inserted 60 simulated query records.")

    cursor.execute("SELECT COUNT(*) FROM query_logs")
    print("Total records in database:", cursor.fetchone()[0])

    connection.close()


if __name__ == "__main__":
    simulate_queries()
