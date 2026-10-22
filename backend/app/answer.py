
from search import search_documents
from reranker import DocumentReranker
from claim_validation import validate_claim
import ollama
import re


# Load the reranker once and reuse it across questions.
reranker = None


def get_reranker():
    global reranker

    if reranker is None:
        reranker = DocumentReranker()

    return reranker


def generate_answer(question: str) -> None:
    """Retrieve, rerank, generate, and validate an answer."""

    if not question or not question.strip():
        print("Please enter a valid question.")
        return

    try:
        # 1. Retrieve candidate chunks.
        results = search_documents(question, top_k=3)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            print(
                "\nAnswer: I couldn't find a supported answer "
                "in the provided documents."
            )
            return

        # 2. Rerank candidates by relevance to the question.
        ranked_results = get_reranker().rerank(
            question=question,
            documents=documents,
            metadatas=[
                metadatas[i] if i < len(metadatas) else {}
                for i in range(len(documents))
            ],
            top_k=3
        )

        if not ranked_results:
            print(
                "\nAnswer: I couldn't find a supported answer "
                "in the provided documents."
            )
            return

        # 3. Assign evidence IDs AFTER reranking.
        evidence = {}

        for i, item in enumerate(ranked_results, start=1):
            metadata = item["metadata"]

            evidence[f"E{i}"] = {
                "text": item["text"],
                "source": metadata.get("source", "Unknown source"),
                "page": metadata.get("page_number", "Unknown page")
            }

        context = "\n\n".join(
            f"[{evidence_id}]\n{item['text']}"
            for evidence_id, item in evidence.items()
        )

        # 4. Generate an answer using the reranked evidence.
        prompt = f"""
You answer questions using company-policy documents.

Rules:
1. Answer directly in a complete sentence.
2. Use only the evidence below.
3. Do not invent facts or add unsupported details.
4. Cite factual statements using evidence IDs, such as [E1].
5. If the evidence does not answer the question, respond exactly:
I couldn't find a supported answer in the provided documents.
6. Only use evidence IDs that appear below.

Evidence:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        answer = response["message"]["content"].strip()

        print("\nDEBUG - Raw model response:")
        print(repr(answer))

        if not answer:
            print("\nAnswer: The model returned an empty response.")
            return

        # 5. Handle unsupported questions.
        if "i couldn't find a supported answer" in answer.lower():
            print("\nAnswer:")
            print(
                "I couldn't find a supported answer "
                "in the provided documents."
            )
            return

        # 6. Extract and validate citation IDs.
        cited_ids = set(re.findall(r"\[E(\d+)\]", answer))

        valid_numbers = {
            evidence_id[1:]
            for evidence_id in evidence
        }

        invalid_ids = cited_ids - valid_numbers

        if invalid_ids:
            print("\nAnswer: Invalid evidence references detected.")
            print("Invalid references:", sorted(invalid_ids))
            return

        if not cited_ids:
            print(
                "\nAnswer: The model did not provide a verifiable "
                "evidence citation. Please retry the question."
            )
            return

        # 7. Validate claims against the cited evidence.
        cited_evidence = "\n".join(
            evidence[f"E{number}"]["text"]
            for number in sorted(cited_ids, key=int)
        )

        answer_without_citations = re.sub(
            r"\[E\d+\]", "", answer
        ).strip()

        claims = [
            claim.strip()
            for claim in re.split(
                r"(?<=[.!?])\s+",
                answer_without_citations
            )
            if claim.strip()
        ]

        if not claims:
            print("\nAnswer: No verifiable claim was found.")
            return

        for claim in claims:
            if not validate_claim(claim, cited_evidence):
                print(
                    "\nAnswer: The generated answer could not "
                    "be validated against the cited evidence."
                )
                print("Unvalidated claim:", claim)
                return

        # 8. Display the validated answer and its sources.
        print("\nAnswer:")
        print(answer)

        print("\nCited sources:")

        for number in sorted(cited_ids, key=int):
            item = evidence[f"E{number}"]

            print(
                f"- [E{number}] {item['source']}, "
                f"page {item['page']}"
            )

    except Exception as error:
        print(f"\nCould not generate an answer: {error}")
        print("Check that Ollama is running and required models are installed.")


if __name__ == "__main__":
    question = input("Ask a policy question: ").strip()

    if question:
        generate_answer(question)
    else:
        print("Please enter a question.")

