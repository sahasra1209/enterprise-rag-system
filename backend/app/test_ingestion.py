
from ingestion import extract_text_from_pdf

pdf_path = "../data/raw/test policy.pdf"

text = extract_text_from_pdf(pdf_path)

print("Extracted PDF text:")
print("-" * 30)
print(text)
