"""EmbeddingRetrieval — embed tool docs + queries, rank tools by similarity.

This is the inference side of the doc-based tool-discovery approach. Given a
corpus of tool documentation, it embeds every tool doc into a vector once, then
for each incoming query embeds the query and returns the top-k most similar tool
slugs by cosine similarity.

Two embedding backends:

``hashing`` (default, zero extra deps)
    Deterministic feature-hashing bag-of-words vectors (numpy only). Captures
    lexical overlap. Useful as a baseline and for tests / CI without GPUs.

``sentence_transformers`` (optional)
    Real semantic embeddings from a sentence-transformers bi-encoder. Install
    with ``pip install trajectory-labs[embedding]``. Use a fine-tuned checkpoint
    (see the ``embedding_sft`` algorithm) by pointing ``model_name`` at it.

The client satisfies the ``InferenceClient`` protocol via ``generate()`` (which
returns the top-1 slug wrapped in ``<answer>...</answer>`` so existing
exact_match/toolkit_match eval still works) and additionally exposes
``retrieve(query, top_k) -> list[str]`` which the eval runner uses to populate
the ``candidates`` field consumed by the ``pass_at_k_retrieval`` metric.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..registry import register_inference

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Embedders — each turns a list[str] into an (n, dim) L2-normalized matrix.
# ---------------------------------------------------------------------------


class _HashingEmbedder:
    """Deterministic feature-hashing bag-of-words embedder (numpy only)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class _SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Optional dependency."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # raises if missing

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Inference client
# ---------------------------------------------------------------------------


class EmbeddingRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = "hashing"
    """'hashing' (zero-dep numpy) or 'sentence_transformers' (optional)."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """For backend=sentence_transformers: HF model name or local checkpoint path."""
    dim: int = 256
    """For backend=hashing: embedding dimension."""
    corpus_path: str | None = None
    """JSONL of tool docs to retrieve over. Each row needs id_field + text_field."""
    corpus_rows: list[dict[str, Any]] | None = None
    """Inline alternative to corpus_path (used by tests / in-memory runs)."""
    text_field: str = "positive"
    """Corpus row field holding the doc text to embed."""
    id_field: str = "tool_slug"
    """Corpus row field holding the candidate label returned by retrieve()."""
    top_k: int = 10
    """Default number of candidates to return."""


@register_inference("embedding_retrieval")
class EmbeddingRetrieval:
    name: ClassVar[str] = "embedding_retrieval"
    Config: ClassVar[type] = EmbeddingRetrievalConfig

    def __init__(
        self,
        *,
        backend: str = "hashing",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 256,
        corpus_path: str | None = None,
        corpus_rows: list[dict[str, Any]] | None = None,
        text_field: str = "positive",
        id_field: str = "tool_slug",
        top_k: int = 10,
    ) -> None:
        self.text_field = text_field
        self.id_field = id_field
        self.top_k = top_k

        if backend == "hashing":
            self.embedder: Any = _HashingEmbedder(dim=dim)
        elif backend == "sentence_transformers":
            self.embedder = _SentenceTransformerEmbedder(model_name)
        else:
            raise ValueError(f"Unknown embedding backend: {backend!r}")

        # Build the tool catalog (dedup by id_field, keep first doc seen).
        rows = self._load_corpus(corpus_path, corpus_rows)
        seen: set[str] = set()
        self.slugs: list[str] = []
        docs: list[str] = []
        for r in rows:
            slug = str(r.get(id_field, "")).strip()
            doc = str(r.get(text_field, "")).strip()
            if not slug or slug in seen:
                continue
            seen.add(slug)
            self.slugs.append(slug)
            docs.append(doc or slug)

        self.doc_matrix = self.embedder.encode(docs) if docs else np.zeros((0, dim), dtype=np.float32)

    @staticmethod
    def _load_corpus(
        corpus_path: str | None,
        corpus_rows: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if corpus_rows is not None:
            return corpus_rows
        if corpus_path:
            p = Path(corpus_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"corpus_path not found: {p}")
            return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        raise ValueError("EmbeddingRetrieval needs either corpus_path or corpus_rows")

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Return the top-k tool slugs ranked by similarity to the query."""
        k = top_k or self.top_k
        if self.doc_matrix.shape[0] == 0:
            return []
        q = self.embedder.encode([query])[0]
        scores = self.doc_matrix @ q  # cosine sim (both L2-normalized)
        order = np.argsort(-scores)[:k]
        return [self.slugs[i] for i in order]

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        top = self.retrieve(prompt, top_k=1)
        answer = top[0] if top else ""
        return f"<answer>{answer}</answer>"
