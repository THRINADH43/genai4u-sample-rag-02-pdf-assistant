"""Answer only from retrieved chunks, cite [file p.N] for every claim, refuse when the
evidence is missing."""

import json
import os
import re
import urllib.request

REFUSAL = "I could not find this in the documents."
MIN_SCORE = 0.30  # below this similarity the best chunk is not about the question

PROMPT = """Answer the question using ONLY the sources below. After every sentence, cite the
source it came from exactly as written, like [leave_policy.pdf p.2]. If the sources do not
contain the answer, reply exactly: {refusal}

Sources:
{sources}

Question: {question}"""


class FakeLLM:
    """Copies the source sentence that shares the most words with the question, with its
    citation. Lets tests run without an API key."""

    def complete(self, prompt):
        question = prompt.rsplit("Question:", 1)[1].lower()
        words = set(re.findall(r"[a-z]{4,}", question))
        best, best_overlap = None, 0
        for line in prompt.split("\n"):
            m = re.match(r"(\[[^\]]+\]) (.*)", line)
            if m:
                for sentence in re.split(r"(?<=\.) ", m.group(2)):
                    overlap = len(words & set(re.findall(r"[a-z]{4,}", sentence.lower())))
                    if overlap > best_overlap:
                        best, best_overlap = f"{sentence} {m.group(1)}", overlap
        return best or REFUSAL


class GroqLLM:
    def __init__(self, model="openai/gpt-oss-20b"):
        self.key = os.environ["GROQ_API_KEY"]
        self.model = model

    def complete(self, prompt):
        body = {"model": self.model, "temperature": 0, "reasoning_effort": "low",
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json", "User-Agent": "pdf-assistant/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()


def check_citations(answer, allowed):
    """Every sentence must end with a citation from the retrieved chunks."""
    if answer.strip() == REFUSAL:
        return True
    sentences = [s for s in re.split(r"(?<=[.\]])\s+(?=[A-Z])", answer.strip()) if s]
    return all(any(c in s for c in allowed) for s in sentences)


def answer(question, index, llm, k=3):
    hits = index.search(question, k)
    hits = [(score, c) for score, c in hits if score >= MIN_SCORE]
    if not hits:
        return REFUSAL, []
    sources = "\n".join(f"{c.cite} {c.text}" for _, c in hits)
    reply = llm.complete(PROMPT.format(refusal=REFUSAL, sources=sources, question=question))
    allowed = [c.cite for _, c in hits]
    if not check_citations(reply, allowed):
        return REFUSAL + " (the model's answer did not cite its sources)", hits
    return reply, hits
