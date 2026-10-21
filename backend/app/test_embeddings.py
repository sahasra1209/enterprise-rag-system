from embeddings import load_embedding_model, generate_embedding

model = load_embedding_model()

text = "Employees are eligible for annual leave."

embedding = generate_embedding(model, text)

print("Original text:", text)
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])