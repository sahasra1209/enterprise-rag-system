
from pathlib import Path
from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    reader = PdfReader(path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

    return pages
