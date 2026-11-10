import unittest

from pdf_assistant.answer import REFUSAL, FakeLLM, answer, check_citations
from pdf_assistant.ingest import Chunk, chunk_text, clean, ingest_folder
from pdf_assistant.search import HashEmbedder, VectorIndex

PAGES = {
    "leave.pdf": ["Every employee gets 24 days of paid leave a year.",
                  "Sick leave is 8 days a year. A doctor's certifi-\ncate is needed after 2 days."],
    "travel.pdf": ["The daily allowance is Rs 1,500 in metro cities."],
}


def fake_reader(path):
    return PAGES[path.split("/")[-1]]


class FakeFolder:
    pass


class IngestTests(unittest.TestCase):
    def test_clean_joins_hyphenated_words_and_spaces(self):
        self.assertEqual(clean("certifi-\ncate   is\n needed"), "certificate is needed")

    def test_chunks_overlap(self):
        words = " ".join(str(i) for i in range(10))
        chunks = chunk_text(words, size=4, overlap=2)
        self.assertEqual(chunks[0], "0 1 2 3")
        self.assertEqual(chunks[1], "2 3 4 5")
        self.assertEqual(chunks[-1].split()[-1], "9")

    def test_page_numbers_are_kept(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for name in PAGES:
                open(os.path.join(d, name), "w").close()
            chunks = ingest_folder(d, reader=fake_reader)
        sick = [c for c in chunks if "Sick" in c.text][0]
        self.assertEqual((sick.file, sick.page), ("leave.pdf", 2))
        self.assertIn("certificate", sick.text)


class AnswerTests(unittest.TestCase):
    def setUp(self):
        chunks = [Chunk(f, p, clean(t)) for f, pages in PAGES.items() for p, t in enumerate(pages, 1)]
        self.index = VectorIndex(HashEmbedder(), chunks)

    def test_answer_cites_file_and_page(self):
        reply, _ = answer("How many days of sick leave?", self.index, FakeLLM())
        self.assertIn("[leave.pdf p.2]", reply)
        self.assertIn("8 days", reply)

    def test_refuses_when_nothing_matches(self):
        reply, hits = answer("Who won the cricket world cup?", self.index, FakeLLM())
        self.assertEqual(reply, REFUSAL)

    def test_uncited_answer_is_rejected(self):
        class Liar:
            def complete(self, prompt):
                return "You get 50 days of sick leave."
        reply, _ = answer("How many days of sick leave?", self.index, Liar())
        self.assertTrue(reply.startswith(REFUSAL))

    def test_check_citations(self):
        allowed = ["[a.pdf p.1]"]
        self.assertTrue(check_citations("It is 8 days. [a.pdf p.1]", allowed))
        self.assertFalse(check_citations("It is 8 days. [b.pdf p.9]", allowed))


if __name__ == "__main__":
    unittest.main()
