
from sentence_transformers import CrossEncoder


class DocumentReranker:
    """Rerank retrieved document chunks against a user query."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        documents: list[str],
        metadatas: list[dict],
        top_k: int = 3
    ) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if len(documents) != len(metadatas):
            raise ValueError(
                "Documents and metadata must have equal lengths."
            )

        if not documents:
            return []

        pairs = [(question, document) for document in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, metadatas, scores),
            key=lambda item: float(item[2]),
            reverse=True
        )

        return [
            {
                "text": document,
                "metadata": metadata,
                "rerank_score": float(score)
            }
            for document, metadata, score in ranked[:top_k]
        ]


if __name__ == "__main__":
    reranker = DocumentReranker()

    sample_question = "Who approves leave requests?"
    sample_documents = [
        "Employees receive 20 days of annual leave per year.",
        "Managers must approve leave requests.",
        "Sick leave is available for eligible employees."
    ]
    sample_metadata = [
        {"source": "policy.pdf", "page_number": 1},
        {"source": "policy.pdf", "page_number": 1},
        {"source": "policy.pdf", "page_number": 1}
    ]

    results = reranker.rerank(
        sample_question,
        sample_documents,
        sample_metadata
    )

    for result in results:
        print(f"Score: {result['rerank_score']:.4f}")
        print(f"Text: {result['text']}\n")
