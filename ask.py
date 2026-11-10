"""python ask.py "How many days of sick leave do I get?"     (uses Groq if GROQ_API_KEY is set,
otherwise the fake model)"""

import os
import sys

from pdf_assistant.answer import FakeLLM, GroqLLM, answer
from pdf_assistant.ingest import ingest_folder
from pdf_assistant.search import MiniLMEmbedder, VectorIndex


def main():
    question = " ".join(sys.argv[1:])
    chunks = ingest_folder("pdfs")
    index = VectorIndex.load_or_build(MiniLMEmbedder(), chunks)  # saved to index.json
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else FakeLLM()
    reply, hits = answer(question, index, llm)
    print(reply)
    print("\nRetrieved:")
    for score, c in hits:
        print(f"  {score:.2f} {c.cite}")


if __name__ == "__main__":
    main()
