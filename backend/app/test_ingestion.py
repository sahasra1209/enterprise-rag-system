
import unittest
from pathlib import Path

from ingestion import extract_pages_from_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "test policy.pdf"


class TestIngestion(unittest.TestCase):

    def test_extract_pages_from_pdf(self):
        self.assertTrue(
            PDF_PATH.exists(),
            f"Test PDF not found: {PDF_PATH}"
        )

        pages = extract_pages_from_pdf(str(PDF_PATH))

        self.assertIsInstance(pages, list)
        self.assertGreater(len(pages), 0)

        for page in pages:
            self.assertIn("page_number", page)
            self.assertIn("text", page)
            self.assertIsInstance(page["page_number"], int)
            self.assertIsInstance(page["text"], str)
            self.assertTrue(page["text"].strip())

    def test_missing_pdf_raises_error(self):
        missing_path = str(
            PDF_PATH.parent / "file_that_does_not_exist.pdf"
        )

        with self.assertRaises(FileNotFoundError):
            extract_pages_from_pdf(missing_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
