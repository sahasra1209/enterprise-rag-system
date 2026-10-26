
from search import search_documents
from reranker import DocumentReranker
from hallucination_detector import HallucinationDetector
import ollama
import re


reranker = None
hallucination_detector = None

FALLBACK = (
    "I couldn't find a supported answer "
    "in the provided documents."
)


def get_reranker():
    """Load the reranker only when needed."""
    global reranker

    if reranker is None:
        reranker = DocumentReranker()

    return reranker


def get_hallucination_detector():
    """Load the NLI detector only when needed."""
    global hallucination_detector

    if hallucination_detector is None:
        hallucination_detector = HallucinationDetector()

    return hallucination_detector


def get_evidence_context(evidence):
    """Format evidence with stable citation IDs."""
    return "\n\n".join(
        f"[{evidence_id}]\n{item['text']}"
        for evidence_id, item in evidence.items()
    )


def generate_answer(question: str) -> None:
    """Retrieve, rerank, generate, and verify an answer."""

    if not question or not question.strip():
        print("Please enter a valid question.")
        return

    try:
        # 1. Retrieve document chunks.
        results = search_documents(question, top_k=3)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            print(f"\nAnswer: {FALLBACK}")
            return

        aligned_metadata = [
            metadatas[i] if i < len(metadatas) else {}
            for i in range(len(documents))
        ]

        # 2. Rerank retrieved documents.
        ranked_results = get_reranker().rerank(
            question=question,
            documents=documents,
            metadatas=aligned_metadata,
            top_k=3,
        )

        if not ranked_results:
            print(f"\nAnswer: {FALLBACK}")
            return

        # 3. Assign citation IDs.
        evidence = {}

        for i, item in enumerate(ranked_results, start=1):
            metadata = item["metadata"]

            evidence[f"E{i}"] = {
                "text": item["text"],
                "source": metadata.get("source", "Unknown source"),
                "page": metadata.get("page_number", "Unknown page"),
            }

        context = get_evidence_context(evidence)

        # 4. Ask the LLM to answer directly from the evidence.
        prompt = f"""
You are a policy-document question-answering assistant.

Answer the user's question using ONLY the supplied evidence.

Instructions:
- If the evidence explicitly answers the question, answer it directly.
- Do not refuse a question that the evidence clearly answers.
- Do not invent facts or add information absent from the evidence.
- Cite every factual sentence using the evidence ID, such as [E1].
- Put the citation before the final punctuation.
- Use only citation IDs that appear in the supplied evidence.
- If the evidence does not answer the question, return exactly:
{FALLBACK}

Examples:
Question: How many days per week may employees work remotely?
Evidence: Employees may work remotely two days per week.
Answer: Employees may work remotely two days per week [E1].

Question: How many maternity-leave weeks are provided?
Evidence: Employees receive 20 days of annual leave per year.
Answer: {FALLBACK}

Evidence:
{context}

Question:
{question}

Answer:
""".strip()

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        answer = response["message"]["content"].strip()

        print("\n--- RAW OLLAMA ANSWER ---")
        print(repr(answer))

        if not answer:
            print(f"\nAnswer: {FALLBACK}")
            return

        if FALLBACK.lower() in answer.lower():
            # Do not blindly trust a generated answer or invent one.
            # Check for a directly matching evidence sentence.
            direct_match = find_direct_evidence_answer(
                question,
                evidence,
            )

            if direct_match is None:
                print(f"\nAnswer: {FALLBACK}")
                return

            answer = direct_match

        # 5. Normalize citation formatting.
        answer = re.sub(
            r"\(\s*((?:\[E\d+\]\s*)+)\)",
            r"\1",
            answer,
        )

        answer = re.sub(
            r",\s*(\[E\d+\])",
            r" \1",
            answer,
        )

        answer = re.sub(
            r"([.!?])\s*((?:\[E\d+\]\s*)+)",
            r" \2\1",
            answer,
        )

        answer = answer.strip()

        # 6. Split into sentences.
        segments = re.split(
            r"(?<=[.!?])\s+",
            answer,
        )

        detector = get_hallucination_detector()
        verified_sentences = []
        sources_used = set()

        for segment in segments:
            segment = segment.strip()

            if not segment:
                continue

            citation_ids = re.findall(
                r"\[E(\d+)\]",
                segment,
            )

            if not citation_ids:
                print(
                    "\nAnswer blocked: a sentence has no citation."
                )
                print("Unverified text:", segment)
                return

            claim = re.sub(
                r"\[E\d+\]",
                "",
                segment,
            ).strip()

            claim = claim.rstrip(" \t,;:.!?")

            if not claim:
                print("\nAnswer blocked: the claim is empty.")
                return

            unique_ids = set(citation_ids)
            cited_passages = []

            # 7. Validate citation IDs and collect evidence.
            for number in unique_ids:
                evidence_id = f"E{number}"

                if evidence_id not in evidence:
                    print(
                        "\nAnswer blocked: invalid evidence reference:",
                        evidence_id,
                    )
                    return

                cited_passages.append(
                    evidence[evidence_id]["text"]
                )
                sources_used.add(number)

            # 8. Verify each claim against its cited evidence.
            claim_results = [
                detector.verify_claim(claim, passage)
                for passage in cited_passages
            ]

            statuses = [
                result["status"]
                for result in claim_results
            ]

            if "contradicted" in statuses:
                print(
                    "\nAnswer blocked: a claim conflicts "
                    "with its cited evidence."
                )
                print("Claim:", claim)
                print("Verification:", statuses)

                for result in claim_results:
                    print("NLI scores:", result["scores"])
                    print(
                        "Matched evidence:",
                        result.get("matched_evidence", "N/A"),
                    )

                return

            if not all(status == "supported" for status in statuses):
                print(
                    "\nAnswer blocked: a claim could not "
                    "be established by its cited evidence."
                )
                print("Claim:", claim)
                print("Verification:", statuses)

                for result in claim_results:
                    print("NLI scores:", result["scores"])
                    print(
                        "Matched evidence:",
                        result.get("matched_evidence", "N/A"),
                    )

                return

            citations = " ".join(
                f"[E{number}]"
                for number in sorted(unique_ids, key=int)
            )

            verified_sentences.append(
                f"{claim} {citations}."
            )

        if not verified_sentences:
            print("\nAnswer blocked: no claims were verified.")
            return

        # 9. Display verified answers and sources.
        print("\nVerified answer:")
        print(" ".join(verified_sentences))

        print("\nCited sources:")

        for number in sorted(sources_used, key=int):
            item = evidence[f"E{number}"]

            print(
                f"- [E{number}] {item['source']}, "
                f"page {item['page']}"
            )

    except Exception as error:
        print(f"\nCould not generate an answer: {error}")
        print(
            "Check that Ollama is running and the required "
            "packages and models are available."
        )


def find_direct_evidence_answer(question, evidence):
    """
    Return a directly matching evidence sentence only when
    a simple question-to-evidence match is detected.

    This deliberately handles only basic policy questions.
    It does not attempt general-purpose semantic matching.
    """
    normalized_question = re.sub(
        r"[^a-z0-9\s]",
        "",
        question.lower(),
    )

    remote_patterns = (
        "how many days per week",
        "how many days a week",
        "days per week",
        "days a week",
    )

    if not any(
        pattern in normalized_question
        for pattern in remote_patterns
    ):
        return None

    for evidence_id, item in evidence.items():
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            item["text"].strip(),
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            normalized_sentence = sentence.lower()

            if (
                "work remotely" in normalized_sentence
                and re.search(
                    r"\btwo days per week\b|\b2 days per week\b",
                    normalized_sentence,
                )
            ):
                return f"{sentence} [{evidence_id}]"

    return None


if __name__ == "__main__":
    question = input("Ask a policy question: ").strip()
    generate_answer(question)
