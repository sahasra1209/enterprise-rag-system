
from embeddings import load_embedding_model
from vector_store import create_vector_store


def search_documents(question: str, top_k: int = 3):
    model = load_embedding_model()
    collection = create_vector_store()

    total_chunks = collection.count()
    print(f"\nTotal chunks in database: {total_chunks}")

    if total_chunks == 0:
        print("The database is empty. Run test_vector_store.py first.")
        return

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"]
    )
    return results

    for i, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][i - 1]
        distance = results["distances"][0][i - 1]

        print(f"\n--- Result {i} ---")
        print(f"Distance: {distance:.4f}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page_number']}")
        print(f"Chunk: {metadata['chunk_number']}")
        print(f"Text: {document}")


if __name__ == "__main__":
    question = input("Ask a question about your PDF: ").strip()

    if question:
        search_documents(question)
    else:
        print("Please enter a question.")
