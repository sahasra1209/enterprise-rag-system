
from ingestion import extract_pages_from_pdf
from chunking import split_text_into_chunks
from embeddings import load_embedding_model
from vector_store import create_vector_store, store_chunks

pdf_path = "data/raw/test policy.pdf"

# Extract PDF text
pages = extract_pages_from_pdf(pdf_path)

# Chunk the text and preserve metadata
all_chunks = []

for page in pages:
    chunks = split_text_into_chunks(
        page["text"],
        chunk_size=100,
        overlap=20
    )

    for chunk_number, chunk_text in enumerate(chunks, start=1):
        all_chunks.append({
            "source": "test policy.pdf",
            "page_number": page["page_number"],
            "chunk_number": chunk_number,
            "text": chunk_text
        })

if not all_chunks:
    raise ValueError("No text chunks found in the PDF.")

# Generate embeddings
model = load_embedding_model()
texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts)

# Save everything in ChromaDB
collection = create_vector_store()
collection.delete(
    ids=collection.get()["ids"]
)
store_chunks(collection, all_chunks, embeddings)

print("Total records in database:", collection.count())
