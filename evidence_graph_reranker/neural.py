"""Optional pretrained neural scorers. Weights remain with their original authors."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from .core import Passage, rank_with_scores

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class EvidenceGraphReranker:
    """Hugging Face-friendly neural scoring + the same fixed graph rule.

    backend='cross-encoder' (the standard MS MARCO reranker) or backend='dense'
    (cosine over normalized all-MiniLM embeddings).

    Models are downloaded from Hugging Face at first inference. Pass scores
    explicitly to `rank` to avoid downloading or loading a neural model.
    """

    def __init__(self, backend: str = "cross-encoder", device: str = "cpu", batch_size: int = 16):
        if backend not in {"cross-encoder", "dense"}:
            raise ValueError("backend must be 'cross-encoder' or 'dense'")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.backend, self.device, self.batch_size = backend, device, batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install the optional dependency: pip install sentence-transformers") from exc
            if self.backend == "cross-encoder":
                self._model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=256, device=self.device)
            else:
                self._model = SentenceTransformer(DENSE_MODEL, device=self.device)
                self._model.max_seq_length = 128
        return self._model

    def score(self, query: str, passages: Sequence[Passage | Mapping[str, Any] | str]) -> np.ndarray:
        docs = [Passage.from_value(p) for p in passages]
        if not docs:
            raise ValueError("At least one passage is required")
        texts = [(p.title + ". " if p.title else "") + p.text for p in docs]
        model = self._get_model()
        if self.backend == "cross-encoder":
            out = model.predict([(query, text) for text in texts],
                                batch_size=self.batch_size, show_progress_bar=False)
            return np.asarray(out, dtype=np.float32).reshape(-1)
        emb = model.encode([query] + texts, batch_size=self.batch_size,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        return (emb[1:] @ emb[0]).astype(np.float32)

    def rank(self, query: str, passages: Sequence[Passage | Mapping[str, Any] | str],
             k: int = 3, scores: Sequence[float] | None = None) -> list[dict[str, Any]]:
        docs = [Passage.from_value(p) for p in passages]
        scores = self.score(query, docs) if scores is None else scores
        return rank_with_scores(query, docs, scores, k=k)
