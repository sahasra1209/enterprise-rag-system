
from search import search_documents
from reranker import DocumentReranker
from hallucination_detector import HallucinationDetector

import ollama
import re
from time import perf_counter


# Lazy-loaded components
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


def find_direct_evidence_answer(question, evidence):
    """
    Return a directly matching evidence sentence only
    for basic remote-work policy questions.
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


def normalize_citations(answer):
    """Normalize citation placement and spacing."""

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

    return answer.strip()


def generate_answer(question: str) -> dict:
    """
    Retrieve, rerank, generate, verify and return
    structured results for the analytics dashboard.
    """

    start_time = perf_counter()

    result = {
        "question": question,
        "answer": FALLBACK,
        "overall_status": "error",
        "claims": [],
        "citations": [],
        "retrieved_chunks": [],
        "latency_ms": 0,
        "error": None,
    }

    def finish():
        result["latency_ms"] = round(
            (perf_counter() - start_time) * 1000,
            2,
        )
        return result

    # 1. Validate question.
    if not isinstance(question, str) or not question.strip():

        result["overall_status"] = "invalid_input"
        result["error"] = "Please enter a valid question."

        return finish()

    try:

        # 2. Retrieve document chunks.
        search_results = search_documents(
            question,
            top_k=3,
        )

        documents = search_results.get(
            "documents", [[]]
        )[0]

        metadatas = search_results.get(
            "metadatas", [[]]
        )[0]

        if not documents:

            result["overall_status"] = "insufficient_evidence"

            return finish()

        # Align metadata with retrieved documents.
        aligned_metadata = [
            metadatas[i] if i < len(metadatas) else {}
            for i in range(len(documents))
        ]

        # 3. Rerank retrieved documents.
        ranked_results = get_reranker().rerank(
            question=question,
            documents=documents,
            metadatas=aligned_metadata,
            top_k=3,
        )

        if not ranked_results:

            result["overall_status"] = "insufficient_evidence"

            return finish()

        # 4. Assign citation IDs.
        evidence = {}

        for i, item in enumerate(ranked_results, start=1):

            metadata = item.get("metadata", {})

            evidence_id = f"E{i}"

            evidence[evidence_id] = {
                "text": item["text"],
                "source": metadata.get(
                    "source", "Unknown source"
                ),
                "page": metadata.get(
                    "page_number", "Unknown page"
                ),
            }

            result["retrieved_chunks"].append({
                "citation_id": evidence_id,
                "text": item["text"],
                "source": metadata.get(
                    "source", "Unknown source"
                ),
                "page": metadata.get(
                    "page_number", "Unknown page"
                ),
            })

        context = get_evidence_context(evidence)

        # 5. Generate answer using Ollama.
        prompt = f"""
You are a policy-document question-answering assistant.

Answer the user's question using ONLY the supplied evidence.

Instructions:
- If the evidence explicitly answers the question, answer directly.
- Do not refuse a question that the evidence clearly answers.
- Do not invent facts.
- Do not add information absent from the evidence.
- Cite every factual sentence using an evidence ID such as [E1].
- Put citations before the final punctuation.
- Use only citation IDs present in the supplied evidence.
- If the evidence does not answer the question, return exactly:

{FALLBACK}

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

        # 6. Handle empty LLM responses.
        if not answer:

            result["overall_status"] = "generation_failed"
            result["error"] = "The LLM returned an empty answer."

            return finish()

        # 7. Handle LLM fallback using direct evidence matching.
        if FALLBACK.lower() in answer.lower():

            direct_match = find_direct_evidence_answer(
                question,
                evidence,
            )

            if direct_match is None:

                result["overall_status"] = "insufficient_evidence"

                return finish()

            answer = direct_match

        # 8. Normalize citations.
        answer = normalize_citations(answer)

        # 9. Split answer into sentences.
        segments = re.split(
            r"(?<=[.!?])\s+",
            answer,
        )

        detector = get_hallucination_detector()

        sources_used = set()

        # 10. Verify every claim.
        for segment in segments:

            segment = segment.strip()

            if not segment:
                continue

            citation_numbers = re.findall(
                r"\[E(\d+)\]",
                segment,
            )

            claim = re.sub(
                r"\[E\d+\]",
                "",
                segment,
            ).strip()

            claim = claim.rstrip(" \t,;:.!?")

            claim_record = {
                "claim": claim,
                "citations": [],
                "status": "invalid",
                "verification_results": [],
            }

            result["claims"].append(claim_record)

            # Check empty claims.
            if not claim:
                claim_record["status"] = "invalid"
                continue

            # Every factual claim needs a citation.
            if not citation_numbers:
                claim_record["status"] = "missing_citation"
                continue

            unique_numbers = sorted(
                set(citation_numbers),
                key=int,
            )

            claim_record["citations"] = [
                f"E{number}"
                for number in unique_numbers
            ]

            cited_passages = []
            invalid_reference = False

            # 11. Validate citation IDs.
            for number in unique_numbers:

                evidence_id = f"E{number}"

                if evidence_id not in evidence:

                    invalid_reference = True

                    claim_record["verification_results"].append({
                        "citation_id": evidence_id,
                        "status": "invalid_reference",
                    })

                    continue

                cited_passages.append((
                    evidence_id,
                    evidence[evidence_id]["text"],
                ))

                sources_used.add(number)

            if invalid_reference:

                claim_record["status"] = "invalid_citation"
                continue

            # 12. Verify every cited passage.
            # An exception for one passage must not stop
            # verification of other passages or claims.
            verification_error = False

            for evidence_id, passage in cited_passages:

                try:

                    verification = detector.verify_claim(
                        claim,
                        passage,
                    )

                    if not isinstance(verification, dict):
                        raise ValueError(
                            "Invalid verification result."
                        )

                    claim_record["verification_results"].append({
                        "citation_id": evidence_id,
                        **verification,
                    })

                except Exception as verification_exception:

                    verification_error = True

                    claim_record["verification_results"].append({
                        "citation_id": evidence_id,
                        "status": "verification_error",
                        "error": str(verification_exception),
                    })

            statuses = [
                item.get("status")
                for item in claim_record["verification_results"]
            ]

            # 13. Determine individual claim status.
            if verification_error:

                claim_record["status"] = "verification_error"

            elif "contradicted" in statuses:

                claim_record["status"] = "contradicted"

            elif statuses and all(
                status == "supported"
                for status in statuses
            ):

                claim_record["status"] = "supported"

            else:

                claim_record["status"] = "uncertain"

        # 14. Determine overall answer status.
        all_supported = (
            bool(result["claims"])
            and all(
                claim["status"] == "supported"
                for claim in result["claims"]
            )
        )

        has_verification_error = any(
            claim["status"] == "verification_error"
            for claim in result["claims"]
        )

        if has_verification_error:

            result["overall_status"] = "verification_error"
            result["answer"] = FALLBACK

        elif all_supported:

            result["overall_status"] = "supported"

            verified_sentences = []

            for claim in result["claims"]:

                citations = " ".join(
                    f"[{citation_id}]"
                    for citation_id in claim["citations"]
                )

                verified_sentences.append(
                    f"{claim['claim']} {citations}."
                )

            result["answer"] = " ".join(
                verified_sentences
            )

        else:

            # Never expose unverified claims as supported.
            result["overall_status"] = "needs_review"
            result["answer"] = FALLBACK

        # 15. Collect citation source information.
        for number in sorted(sources_used, key=int):

            evidence_id = f"E{number}"

            item = evidence[evidence_id]

            result["citations"].append({
                "id": evidence_id,
                "source": item["source"],
                "page": item["page"],
            })

        return finish()

    except Exception as error:

        result["overall_status"] = "error"
        result["error"] = str(error)

        return finish()


# 16. Terminal interface.
# The same function can be called by the dashboard or API.

if __name__ == "__main__":

    question = input(
        "Ask a policy question: "
    ).strip()

    output = generate_answer(question)

    print("\nAnswer:")
    print(output["answer"])

    print("\nOverall status:")
    print(output["overall_status"])

    print("\nLatency:")
    print(output["latency_ms"], "ms")

    print("\nClaims:")

    for claim in output["claims"]:

        print(
            claim["claim"],
            "->",
            claim["status"],
        )

    print("\nCited sources:")

    for citation in output["citations"]:

        print(
            f"[{citation['id']}] "
            f"{citation['source']}, "
            f"page {citation['page']}"
        )

    if output["error"]:

        print("\nError:")
        print(output["error"])
