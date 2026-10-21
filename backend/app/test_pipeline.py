
from ingestion import extract_pages_from_pdf
from chunking import split_text_into_chunks
from embeddings import load_embedding_model

# Step 1: Extract pages from the PDF
pdf_path = "data/raw/test policy.pdf"
pages = extract_pages_from_pdf(pdf_path)

# Step 2: Split each page into chunks and preserve metadata
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

# Step 3: Load the embedding model
model = load_embedding_model()

# Step 4: Generate one embedding per chunk
texts = [chunk["text"] for chunk in all_chunks]
vectors = model.encode(texts)

# Step 5: Display results
print("Total pages:", len(pages))
print("Total chunks:", len(all_chunks))
print("Number of embeddings:", len(vectors))

if len(vectors) > 0:
    print("Dimensions per embedding:", len(vectors[0]))
    print("\nFirst chunk metadata:")
    print(all_chunks[0]["source"])
    print(all_chunks[0]["page_number"])
    print(all_chunks[0]["chunk_number"])
    print("\nFirst 10 embedding values:")
    print(vectors[0][:10].tolist())
