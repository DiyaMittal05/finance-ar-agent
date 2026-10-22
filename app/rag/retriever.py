"""Retrieval over policy and contract documents.

BM25 over heading-scoped chunks. No embedding API is needed, so it runs
offline and deterministically; swapping in a vector store only means
replacing `Retriever.search`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import settings

STOPWORDS = set("""a an and are as at be by for from has have if in into is it its of on or
that the their this to was we what when which will with do does our us you your""".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:/[0-9]+)?", text.lower())
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w not in STOPWORDS]


@dataclass
class Chunk:
    chunk_id: str   # e.g. "collections_policy.md#stage-2-31-60-days-overdue"
    source: str
    heading: str
    text: str


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def chunk_markdown(path: Path, root: Path) -> list[Chunk]:
    """One chunk per section. The document title is prefixed to every chunk for context."""
    source = str(path.relative_to(root))
    lines = path.read_text().splitlines()
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), source)
    chunks, heading, buf = [], title, []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(f"{source}#{_slug(heading)}", source, heading,
                                f"[{title} › {heading}]\n{body}"))

    for line in lines:
        if re.match(r"^#{1,4} ", line):
            flush()
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


class Retriever:
    def __init__(self, docs_dir: Path, k1: float = 1.4, b: float = 0.75):
        self.chunks = [c for p in sorted(docs_dir.rglob("*.md")) for c in chunk_markdown(p, docs_dir)]
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [{"chunk_id": self.chunks[i].chunk_id, "score": round(s, 3), "text": self.chunks[i].text}
                for s, i in scores[:k]]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(settings.docs_dir)
