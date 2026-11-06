import sys

sys.path.insert(0, "backend/app")

from search import search_documents
from reranker import DocumentReranker


# One initial evaluation case.
# We'll add more questions after this runs successfully.
TEST_CASES = [
    {
        "question": "What was 3M worldwide net sales in 2015, in millions?",
        "expected_terms": ["net sales", "worldwide", "30,274"],
    }
]

TOP_K = 20


def is_relevant(document, expected_terms):
    """Return True if the document contains all expected evidence terms."""
    document = document.lower()
    return all(term.lower() in document for term in expected_terms)


def find_rank(documents, expected_terms):
    """Return the rank of the first relevant document, or None if not found."""
    for rank, document in enumerate(documents, start=1):
        if is_relevant(document, expected_terms):
            return rank

    return None


def reciprocal_rank(rank):
    """MRR contribution: 1/rank, or 0 if evidence was not found."""
    return 1 / rank if rank else 0


def main():
    reranker = DocumentReranker()

    for case in TEST_CASES:
        question = case["question"]
        expected_terms = case["expected_terms"]

        print("\n" + "=" * 60)
        print("Question:", question)
        print("=" * 60)

        # Step 1: Retrieve candidate documents.
        results = search_documents(question, top_k=TOP_K)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Step 2: Measure retrieval ranking before reranking.
        retrieval_rank = find_rank(documents, expected_terms)

        print("\nBEFORE RERANKING")
        print("Retrieved candidates:", len(documents))
        print("Relevant evidence rank:", retrieval_rank)

        for k in [5, 10, 20]:
            recall = int(
                retrieval_rank is not None and retrieval_rank <= k
            )
            print(f"Recall@{k}: {recall}")

        print("MRR:", round(reciprocal_rank(retrieval_rank), 4))

        # Step 3: Rerank the retrieved candidates.
        reranked_results = reranker.rerank(
            question=question,
            documents=documents,
            metadatas=metadatas,
            top_k=TOP_K,
        )

        reranked_documents = [
            item["text"] for item in reranked_results
        ]

        reranked_rank = find_rank(reranked_documents, expected_terms)

        print("\nAFTER RERANKING")
        print("Relevant evidence rank:", reranked_rank)
        print("MRR:", round(reciprocal_rank(reranked_rank), 4))

        if reranked_rank is not None:
            print("\nRelevant evidence found:")
            print(reranked_results[reranked_rank - 1]["text"])
        else:
            print("\nRelevant evidence was not found after reranking.")

        print("\nTop 5 reranked results:")
        for rank, item in enumerate(reranked_results[:5], start=1):
            print(f"\nRank {rank} | Score: {item['rerank_score']:.4f}")
            print(item["text"][:500])


if __name__ == "__main__":
    main()