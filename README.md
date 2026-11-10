# PDF assistant with citations

Answers questions about a folder of PDFs and cites the file and page of each claim, like
`[leave_policy.pdf p.2]`. It answers only from retrieved text and says so when the answer
is not in the PDFs.

## Setup and run

```
pip install -r requirements.txt     # pypdf, sentence-transformers (CPU is fine), fpdf2
python make_sample_pdfs.py          # creates the two sample PDFs in pdfs/ (already committed)
cp .env.example .env              # or export GROQ_API_KEY=...; never commit the key
GROQ_API_KEY=... python ask.py "How many days of sick leave do I get?"
python ask.py "..."                 # without a key: the fake model, so it still runs
python -m unittest discover -s tests -t .
```

Real output (Groq, `openai/gpt-oss-20b`):

```
$ python ask.py "How many days of sick leave do I get?"
You get 8 days of sick leave a year. [leave_policy.pdf p.2]
Retrieved:
  0.78 [leave_policy.pdf p.2]
  0.59 [leave_policy.pdf p.1]

$ python ask.py "What is the hotel limit per night?"
The hotel limit is Rs 4,000 per night with a receipt. [travel_policy.pdf p.2]

$ python ask.py "Who is the CEO of the company?"
I could not find this in the documents.
```

## How it works

1. **Extract** (`pdf_assistant/ingest.py`): `pypdf` reads each page separately.
2. **Clean:** remove repeated headers and footers (a first or last line found on more than half
   of a file's pages, such as "GenAI4U Sample Co. - HR policies" and "Internal use only" in
   the sample PDFs), remove soft hyphens, join words split across lines (`certifi-\ncate`),
   and collapse spaces.
3. **Chunk:** 60 words with 15 words of overlap, so a sentence cut at a boundary still appears
   whole in one chunk. Chunks never cross a page, so each has exactly one correct page number.
4. **Search** (`pdf_assistant/search.py`): embeddings from `all-MiniLM-L6-v2`, a small local
   model (80 MB, runs on CPU, no API). Vectors are normalised, so cosine is a dot product.
   Ties are broken by file and page, so results repeat exactly. The vectors are saved to
   `index.json` the first time and reused after that; the index is rebuilt only when the
   chunks change. `index.json` is in `.gitignore`.
5. **Answer** (`pdf_assistant/answer.py`):
   - If the best chunk scores below 0.30 similarity, refuse without calling the model.
   - Otherwise the prompt gives only the top 3 chunks, each labelled with its citation, and
     tells the model to cite after every sentence or reply with the refusal sentence.
   - The reply is checked in code: every sentence must contain a citation from the retrieved
     chunks. If not, the user gets the refusal instead of an uncited answer.

## Tests (no API key, no model download)

`FakeLLM` answers with the source sentence that best matches the question, and
`HashEmbedder` is a word-hash embedder, so tests are fast and offline. They cover cleaning,
header and footer removal, building the index once and reusing it,
chunk overlap, page numbers, a cited answer, refusal when nothing matches, rejecting an
uncited answer, and the citation checker.

## Limits

Scanned PDFs (images) have no text layer, so `pypdf` returns nothing for them; they would
need OCR. Tables come out as plain text and can lose their structure.
