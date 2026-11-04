
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# Database location
DB_PATH = Path(__file__).resolve().parent / "analytics.db"


# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data(ttl=10)
def load_data():
    """Load query logs and daily analytics from SQLite."""

    connection = sqlite3.connect(DB_PATH)

    query_logs = pd.read_sql_query(
        "SELECT * FROM query_logs ORDER BY created_at DESC",
        connection
    )

    daily_summary = pd.read_sql_query(
        "SELECT * FROM daily_summary ORDER BY report_date",
        connection
    )

    connection.close()

    return query_logs, daily_summary


# Dashboard heading
st.title("📊 Enterprise RAG Monitoring Dashboard")
st.write("Monitor query activity, answer quality, response time and simulated cost.")

# Refresh button
if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()


# Load database information
try:
    logs, daily = load_data()
except Exception as error:
    st.error(f"Could not load database: {error}")
    st.stop()


if logs.empty:
    st.warning("No API query records found. Ask a question through the API first.")
    st.stop()


# Convert date column
logs["created_at"] = pd.to_datetime(logs["created_at"])
logs["query_date"] = logs["created_at"].dt.date


# Calculate overall metrics
total_queries = len(logs)
average_confidence = logs["confidence_score"].mean()
average_response_time = logs["response_time_ms"].mean()
total_cost = logs["simulated_cost_usd"].sum()


# KPI cards
st.subheader("Overall Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Queries", total_queries)
col2.metric("Average Confidence", f"{average_confidence:.2f}")
col3.metric("Average Response Time", f"{average_response_time:.2f} ms")
col4.metric("Total Simulated Cost", f"${total_cost:.6f}")

# Confidence-based alerting
st.divider()
st.subheader("🚨 Confidence Monitoring Alert")

ALERT_THRESHOLD = 0.70
ROLLING_WINDOW = 10

# Logs are already sorted newest first.
recent_queries = logs.head(ROLLING_WINDOW)

rolling_confidence = recent_queries[
    "confidence_score"
].mean()

st.metric(
    "Latest 10 Queries - Average Confidence",
    f"{rolling_confidence:.2f}"
)

st.caption(
    f"Alert threshold: {ALERT_THRESHOLD:.2f} | "
    f"Queries monitored: {len(recent_queries)}"
)

if len(recent_queries) < ROLLING_WINDOW:
    st.info(
        f"Only {len(recent_queries)} queries available. "
        "The alert will use the available records."
    )

if rolling_confidence < ALERT_THRESHOLD:
    st.error(
        "⚠️ LOW CONFIDENCE ALERT! "
        "The recent average confidence is below 0.70. "
        "Please investigate RAG answer quality."
    )
else:
    st.success(
        "✅ SYSTEM HEALTHY: Recent average confidence "
        "is above the alert threshold."
    )



st.divider()


# Daily query volume
st.subheader("📈 Daily Query Volume")

if not daily.empty:
    st.line_chart(
        daily.set_index("report_date")["total_queries"]
    )
else:
    st.info("Daily summary is not available yet.")


# Daily confidence and response time
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confidence Trend")

    if not daily.empty:
        st.line_chart(
            daily.set_index("report_date")["average_confidence"]
        )

with col2:
    st.subheader("Response Time Trend")

    if not daily.empty:
        st.line_chart(
            daily.set_index("report_date")["average_response_time_ms"]
        )


st.divider()


# Cost per query
st.subheader("💰 Cost Per Query")

cost_data = logs[
    ["id", "question", "estimated_token_count", "simulated_cost_usd"]
].copy()

cost_data["simulated_cost_usd"] = cost_data[
    "simulated_cost_usd"
].round(8)

st.dataframe(cost_data, use_container_width=True)


# Latency distribution
st.subheader("⏱️ Latency Distribution")

latency_data = logs[["response_time_ms"]].copy()

latency_data["latency_range"] = pd.cut(
    latency_data["response_time_ms"],
    bins=[0, 1000, 5000, 10000, 30000, float("inf")],
    labels=[
        "0–1 sec",
        "1–5 sec",
        "5–10 sec",
        "10–30 sec",
        "Above 30 sec"
    ],
    include_lowest=True
)

latency_counts = (
    latency_data["latency_range"]
    .value_counts(sort=False)
    .rename_axis("Latency Range")
    .reset_index(name="Number of Queries")
)

st.bar_chart(
    latency_counts.set_index("Latency Range")
)


# Simple keyword-based question categorization
def classify_category(question):
    question = str(question).lower()

    if any(word in question for word in ["revenue", "profit", "capital", "cost", "expense", "debt"]):
        return "Finance"

    if any(word in question for word in ["employee", "staff", "salary", "workforce", "hiring"]):
        return "Human Resources"

    if any(word in question for word in ["product", "manufacturing", "production", "supply"]):
        return "Operations"

    return "General"


# Failed and review-needed questions
st.subheader("⚠️ Failing Questions by Category")

failed_logs = logs[
    logs["status"].fillna("").str.lower() != "supported"
].copy()

if not failed_logs.empty:
    failed_logs["category"] = failed_logs["question"].apply(
        classify_category
    )

    category_counts = (
        failed_logs.groupby("category")
        .size()
        .reset_index(name="Failed Queries")
        .sort_values("Failed Queries", ascending=False)
    )

    st.bar_chart(
        category_counts.set_index("category")
    )

    st.dataframe(
        failed_logs[["question", "status", "category"]],
        use_container_width=True
    )
else:
    st.success("No failing or review-needed questions recorded yet.")


# Complete query history
st.divider()
st.subheader("📋 Query History")

st.dataframe(
    logs[
        [
            "id",
            "created_at",
            "question",
            "status",
            "confidence_score",
            "response_time_ms",
            "simulated_cost_usd"
        ]
    ],
    use_container_width=True
)


st.caption(
    "Note: Confidence is a status-based monitoring score. "
    "Question categories use simple keyword rules. "
    "Token counts and costs are simulated estimates."
)
