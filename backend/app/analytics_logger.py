import sqlite3
import json
from pathlib import Path
from datetime import datetime


# Database will be created inside backend/app
DB_PATH = Path(__file__).resolve().parent / "analytics.db"


def init_db():
    """Create the analytics database and table if they do not exist."""

    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            retrieved_chunks TEXT,
            answer TEXT,
            status TEXT,
            confidence_score REAL,
            response_time_ms REAL,
            estimated_token_count INTEGER,
            simulated_cost_usd REAL
        )
    """)

    connection.commit()
    connection.close()


def calculate_confidence(status):
    """
    This is a simple status-based monitoring score,
    not a model-generated probability.
    """

    scores = {
        "supported": 1.0,
        "needs_review": 0.5,
        "insufficient_evidence": 0.0,
        "invalid_input": 0.0,
        "generation_failed": 0.0,
        "verification_error": 0.0,
        "error": 0.0
    }

    return scores.get(status, 0.0)


def log_request(result):
    """Save one API question and its result in SQLite."""

    question = result.get("question", "")
    answer = result.get("answer", "")
    status = result.get("overall_status", "error")
    retrieved_chunks = result.get("retrieved_chunks", [])
    response_time_ms = result.get("latency_ms", 0.0)

    # Approximate token count using characters / 4.
    text_to_measure = question + " " + answer
    estimated_token_count = max(1, len(text_to_measure) // 4)

    # Illustrative simulated rate: $0.002 per 1,000 tokens.
    simulated_cost_usd = (
        estimated_token_count / 1000
    ) * 0.002

    confidence_score = calculate_confidence(status)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
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
    """, (
        datetime.now().isoformat(timespec="seconds"),
        question,
        json.dumps(retrieved_chunks),
        answer,
        status,
        confidence_score,
        response_time_ms,
        estimated_token_count,
        simulated_cost_usd
    ))

    connection.commit()
    connection.close()