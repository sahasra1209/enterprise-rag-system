
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "analytics.db"


def run_etl():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # Create a table for daily analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            report_date TEXT PRIMARY KEY,
            total_queries INTEGER,
            average_confidence REAL,
            average_response_time_ms REAL,
            total_simulated_cost_usd REAL
        )
    """)

    # Calculate daily metrics from query logs
    cursor.execute("""
        SELECT
            DATE(created_at),
            COUNT(*),
            AVG(confidence_score),
            AVG(response_time_ms),
            SUM(simulated_cost_usd)
        FROM query_logs
        GROUP BY DATE(created_at)
    """)

    daily_data = cursor.fetchall()

    # Store the calculated metrics
    for row in daily_data:
        cursor.execute("""
            INSERT OR REPLACE INTO daily_summary (
                report_date,
                total_queries,
                average_confidence,
                average_response_time_ms,
                total_simulated_cost_usd
            )
            VALUES (?, ?, ?, ?, ?)
        """, row)

    connection.commit()

    print("Daily analytics summary:")
    for row in daily_data:
        print(row)

    connection.close()


if __name__ == "__main__":
    run_etl()
