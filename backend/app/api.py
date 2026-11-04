
from fastapi import FastAPI
from pydantic import BaseModel
from answer import generate_answer
from analytics_logger import init_db, log_request

app = FastAPI(
    title="Enterprise RAG API",
    description="API for asking questions from enterprise documents",
    version="1.0.0"
)

# Make sure the database and table exist
init_db()


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Enterprise RAG API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: AskRequest):
    result = generate_answer(request.question)

    # Save the request and response in SQLite
    try:
        log_request(result)
        print("Analytics: request saved successfully!")
    except Exception as error:
        # Logging errors should not stop the RAG API
        print(f"Analytics logging failed: {error}")

    return result
