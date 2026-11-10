"""Embedding search. The real embedder is a small local model (all-MiniLM-L6-v2, 80 MB,
runs on CPU). Tests use HashEmbedder, which needs no download."""

import hashlib
import json
import math
import os
import re


class MiniLMEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name, device="cpu")

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class HashEmbedder:
    """Bag of words hashed into a fixed-size vector. Deterministic, for tests only."""

    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, texts):
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # vectors are already normalised


class VectorIndex:
    def __init__(self, embedder, chunks, vectors=None):
        self.embedder = embedder
        self.chunks = chunks
        if vectors is None:
            vectors = embedder.embed([c.text for c in chunks]) if chunks else []
        self.vectors = vectors

    @classmethod
    def load_or_build(cls, embedder, chunks, path="index.json"):
        """Embedding is the slow part, so the vectors are saved once and reused. The saved index
        is rebuilt only when the chunks change (a PDF was added, changed or removed)."""
        key = hashlib.sha256(json.dumps([[c.file, c.page, c.text] for c in chunks]).encode()).hexdigest()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("key") == key:
                return cls(embedder, chunks, saved["vectors"])
        index = cls(embedder, chunks)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"key": key, "vectors": index.vectors}, f)
        return index

    def search(self, query, k=3):
        q = self.embedder.embed([query])[0]
        scored = [(cosine(q, v), c) for v, c in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda sc: (-sc[0], sc[1].file, sc[1].page))  # stable order for ties
        return scored[:k]
