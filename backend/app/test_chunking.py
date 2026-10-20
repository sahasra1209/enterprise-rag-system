
from ingestion import extract_pages_from_pdf
from chunking import split_text_into_chunks

pdf_path = "data/raw/test policy.pdf"

pages = extract_pages_from_pdf(pdf_path)

all_chunks = []

for page in pages:
    chunks = split_text_into_chunks(
        page["text"],
        chunk_size=500,
        overlap=100
    )

    for chunk_number, chunk_text in enumerate(chunks, start=1):
        all_chunks.append({
            "source": "test policy.pdf",
            "page_number": page["page_number"],
            "chunk_number": chunk_number,
            "text": chunk_text
        })

print("Total pages:", len(pages))
print("Total chunks:", len(all_chunks))

for chunk in all_chunks:
    print("\n--- Chunk ---")
    print("Source:", chunk["source"])
    print("Page:", chunk["page_number"])
    print("Chunk number:", chunk["chunk_number"])
    print("Text:", chunk["text"])
