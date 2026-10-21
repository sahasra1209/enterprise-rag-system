
from search import search_documents


def generate_answer(question: str) -> None:
    """Answer policy questions using retrieved document text."""

    results = search_documents(question, top_k=3)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        print("\nNo policy information was found.")
        return

    context = " ".join(documents).lower()
    q = question.lower()

    answer = None

    if "annual leave" in q and "how many days" in q:
        if "20 days of annual leave" in context:
            answer = "Employees receive 20 days of annual leave per year."

    elif "who" in q and "leave" in q and "approve" in q:
        if "managers must approve leave requests" in context:
            answer = "Managers must approve leave requests."

    elif "where" in q and "leave" in q and "request" in q:
        if "hr portal" in context:
            answer = "Employees can request leave through the HR portal."

    if answer is None:
        print("\nAnswer: I couldn't find a supported answer in the retrieved text.")
    else:
        print("\nAnswer:", answer)

    print("\nSources:")
    seen = set()

    for metadata in metadatas:
        source = (metadata["source"], metadata["page_number"])

        if source not in seen:
            seen.add(source)
            print(f"- {source[0]}, page {source[1]}")


if __name__ == "__main__":
    question = input("Ask a policy question: ").strip()

    if question:
        generate_answer(question)
    else:
        print("Please enter a question.")
