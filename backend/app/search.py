
from embeddings import load_embedding_model
from vector_store import create_vector_store


def search_documents(
    question: str,
    top_k: int = 3,
    max_distance: float | None = None
) -> dict:
    """Retrieve document chunks and return their text, metadata, and distances.

    Set max_distance to apply an optional distance cutoff.
    Smaller distances are closer matches for the configured metric,
    but distance alone does not prove that a chunk is relevant.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance cannot be negative.")

    empty_result = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }

    model = load_embedding_model()
    collection = create_vector_store()

    total_chunks = collection.count()
    print(f"\nTotal chunks in database: {total_chunks}")

    if total_chunks == 0:
        print("The database is empty. Run test_vector_store.py first.")
        return empty_result

    question_embedding = model.encode(
        question.strip()
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"]
    )

    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    candidates = list(zip(documents, metadatas, distances))

    if max_distance is not None:
        candidates = [
            (doc, meta, distance)
            for doc, meta, distance in candidates
            if distance <= max_distance
        ]

    print(f"Chunks returned: {len(documents)}")
    print(f"Chunks retained: {len(candidates)}")

    for i, (document, metadata, distance) in enumerate(
        candidates, start=1
    ):
        print(f"\n--- Retrieved Result {i} ---")
        print(f"Distance: {distance:.4f}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Page: {metadata.get('page_number', 'Unknown')}")
        print(f"Chunk: {metadata.get('chunk_number', 'Unknown')}")
        print(f"Text: {document}")

    if not candidates:
        print("No chunks matched the configured distance cutoff.")

    return {
        "documents": [[item[0] for item in candidates]],
        "metadatas": [[item[1] for item in candidates]],
        "distances": [[item[2] for item in candidates]]
    }


if __name__ == "__main__":
    question = input("Ask a question about your PDF: ").strip()

    if question:
        search_documents(question)
    else:
        print("Please enter a question.")
