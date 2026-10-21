
import chromadb


def create_vector_store():
    """Create or load a persistent ChromaDB collection."""

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="policy_documents"
    )

    return collection


def store_chunks(collection, chunks, embeddings):
    """Store chunk text, vectors, and metadata."""

    for i, chunk in enumerate(chunks):
        collection.upsert(
            ids=[f"chunk_{i + 1}"],
            documents=[chunk["text"]],
            embeddings=[embeddings[i].tolist()],
            metadatas=[{
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_number": chunk["chunk_number"]
            }]
        )

    print(f"Stored {len(chunks)} chunks successfully!")
