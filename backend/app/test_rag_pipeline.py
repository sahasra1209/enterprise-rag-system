
import unittest
from unittest.mock import patch

from answer import generate_answer


class TestRAGPipeline(unittest.TestCase):

    @patch("builtins.print")
    @patch("answer.ollama.chat")
    @patch("answer.search_documents")
    @patch("answer.get_reranker")
    def test_supported_question(
        self,
        mock_get_reranker,
        mock_search,
        mock_chat,
        mock_print
    ):
        mock_search.return_value = {
            "documents": [[
                "Managers must approve leave requests."
            ]],
            "metadatas": [[{
                "source": "test policy.pdf",
                "page_number": 1
            }]]
        }

        mock_get_reranker.return_value.rerank.return_value = [{
            "text": "Managers must approve leave requests.",
            "metadata": {
                "source": "test policy.pdf",
                "page_number": 1
            },
            "rerank_score": 7.5
        }]

        mock_chat.return_value = {
            "message": {
                "content": "Managers must approve leave requests. [E1]"
            }
        }

        generate_answer("Who must approve leave requests?")

        printed = " ".join(
            str(call) for call in mock_print.call_args_list
        )
        self.assertIn("Managers must approve leave requests.", printed)
        self.assertIn("Cited sources:", printed)

    @patch("builtins.print")
    @patch("answer.ollama.chat")
    @patch("answer.search_documents")
    @patch("answer.get_reranker")
    def test_unsupported_question(
        self,
        mock_get_reranker,
        mock_search,
        mock_chat,
        mock_print
    ):
        mock_search.return_value = {
            "documents": [[
                "Managers must approve leave requests."
            ]],
            "metadatas": [[{
                "source": "test policy.pdf",
                "page_number": 1
            }]]
        }

        mock_get_reranker.return_value.rerank.return_value = [{
            "text": "Managers must approve leave requests.",
            "metadata": {
                "source": "test policy.pdf",
                "page_number": 1
            },
            "rerank_score": 7.5
        }]

        mock_chat.return_value = {
            "message": {
                "content": (
                    "I couldn't find a supported answer "
                    "in the provided documents."
                )
            }
        }

        generate_answer("How many maternity-leave weeks are provided?")

        printed = " ".join(
            str(call) for call in mock_print.call_args_list
        )
        self.assertIn("couldn't find a supported answer", printed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
