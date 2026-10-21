from sentence_transformers import SentenceTransformer


def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def generate_embedding(model, text: str) -> list[float]:
    embedding = model.encode(text)
    return embedding.tolist()