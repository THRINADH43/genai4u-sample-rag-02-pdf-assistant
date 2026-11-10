"""Extract text from PDFs page by page, clean it, and cut it into chunks that remember
their file and page."""

import os
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    file: str
    page: int  # 1-based, as people count pages
    text: str

    @property
    def cite(self):
        return f"[{self.file} p.{self.page}]"


def clean(text):
    text = text.replace("\u00ad", "")              # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)          # words split across lines: "employ-\nee"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_repeated_lines(pages):
    """Remove header and footer lines: a first or last line that appears on more than half of
    a file's pages (like "Acme Ltd - Leave policy" or "Company confidential")."""
    if len(pages) < 2:
        return pages
    counts = {}
    for page in pages:
        lines = [l.strip() for l in page.splitlines() if l.strip()]
        for line in set(lines[:1] + lines[-1:]):
            counts[line] = counts.get(line, 0) + 1
    repeated = {line for line, n in counts.items() if n > len(pages) / 2}
    return ["\n".join(l for l in page.splitlines() if l.strip() not in repeated) for page in pages]


def chunk_text(text, size=60, overlap=15):
    """Split into chunks of `size` words with `overlap` words shared between neighbours,
    so a sentence cut at a boundary still appears whole in one chunk."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def read_pdf_pages(path):
    from pypdf import PdfReader  # imported here so tests without pypdf still run
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def ingest_folder(folder, reader=read_pdf_pages, size=60, overlap=15):
    """Chunks never cross a page, so every chunk has exactly one correct page number."""
    chunks = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".pdf"):
            continue
        pages = strip_repeated_lines(reader(os.path.join(folder, name)))
        for page_no, raw in enumerate(pages, start=1):
            text = clean(raw)
            for piece in chunk_text(text, size, overlap) if text else []:
                chunks.append(Chunk(name, page_no, piece))
    return chunks
