from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str)->str:
    path=Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"The file {pdf_path} does not exist.") 
    reader=PdfReader(path)
    extracted_pages=[]
    for page in reader.pages:
        text=page.extract_text()
        if text:
            extracted_pages.append(text)
    return "\n".join(extracted_pages)