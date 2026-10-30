"""The *single* fixed graph selector used across all six reported benchmarks.

Only candidate text, its optional title, and a query-specific neural relevance
score are allowed as input. There is no training, gold-label access, or corpus-
specific parameter tuning inside this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import re
import unicodedata
import numpy as np


def normalize_text(text: Any) -> str:
    """Match the exact normalization of the published evaluation scripts."""
    return " ".join(
        re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKD", str(text)).casefold()).split()
    )


@dataclass(frozen=True)
class Passage:
    title: str
    text: str
    id: str | None = None

    @classmethod
    def from_value(cls, value: "Passage | Mapping[str, Any] | str") -> "Passage":
        if isinstance(value, Passage):
            return value
        if isinstance(value, str):
            return cls(title="", text=value)
        if not isinstance(value, Mapping):
            raise TypeError("Each passage must be a string, dict, or Passage.")
        text = value.get("text", value.get("body", value.get("paragraph_text", "")))
        return cls(title=str(value.get("title", "") or ""), text=str(text or ""),
                   id=None if value.get("id") is None else str(value["id"]))


def graph_features(query: str, passages: Sequence[Passage]) -> tuple[np.ndarray, np.ndarray]:
    """Return directed source->target article-title edges and query/title matches.

    This is intentionally the SAME simple string-link rule as the six-dataset
    benchmarks. In particular, it does NOT introduce token overlap, semantic
    similarity edges, facet decomposition, or learned relation extraction.
    """
    q = normalize_text(query)
    titles = [normalize_text(p.title) for p in passages]
    bodies = [normalize_text(p.text) for p in passages]
    q_words = set(q.split())
    n = len(passages)
    link = np.zeros((n, n), np.float32)
    match = np.zeros(n, np.float32)
    for j, title in enumerate(titles):
        if not title:
            continue
        title_words = set(title.split())
        overlap = len(title_words & q_words) / max(1, len(title_words))
        match[j] = 1.0 if " " + title + " " in " " + q + " " else 0.5 * overlap
        for i, body in enumerate(bodies):
            if i != j and len(title) >= 4 and " " + title + " " in " " + body + " ":
                link[i, j] = 1.0
    return link, match


def minmax_relevance(scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a one-dimensional array of finite numbers.")
    if not len(scores):
        raise ValueError("At least one candidate passage is required.")
    lo, hi = float(scores.min()), float(scores.max())
    return (scores - lo) / (hi - lo) if hi > lo + 1e-9 else np.zeros_like(scores)


def select_with_features(
    scores: Sequence[float], link: np.ndarray, match: np.ndarray, k: int = 3
) -> list[int]:
    """Greedy rank with fixed coefficients .4, .5, .2 and deterministic ties.

    At step t the score of candidate j is
      0.4 * normalized_relevance_j
      + 0.5 * max_{i previously selected} link[i, j]
      + 0.2 * query_title_match_j.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    norm_scores = minmax_relevance(scores)
    n = len(norm_scores)
    link = np.asarray(link, dtype=np.float32)
    match = np.asarray(match, dtype=np.float32)
    if link.shape != (n, n) or match.shape != (n,):
        raise ValueError("Link matrix or match vector dimensions do not match scores")
    selected: list[int] = []
    remaining = set(range(n))
    for _ in range(min(k, n)):
        nxt = max(remaining, key=lambda j: (
            0.4 * norm_scores[j]
            + 0.5 * max((link[i, j] for i in selected), default=0.0)
            + 0.2 * match[j],
            -j,
        ))
        selected.append(nxt)
        remaining.remove(nxt)
    return selected


def rank_with_scores(query: str, passages: Sequence[Passage | Mapping[str, Any] | str],
                     scores: Sequence[float], k: int = 3) -> list[dict[str, Any]]:
    """Public, network-free inference API if the retriever supplies scores.

    Returned `step_score` is the acquisition score at the moment of selection,
    not a calibrated relevance probability.
    """
    docs = [Passage.from_value(p) for p in passages]
    if len(docs) != len(scores):
        raise ValueError("Supply exactly one relevance score per candidate passage")
    link, match = graph_features(query, docs)
    normalized = minmax_relevance(scores)
    order = select_with_features(scores, link, match, k)
    result = []
    selected = []
    for idx in order:
        bonus = max((float(link[i, idx]) for i in selected), default=0.0)
        step = 0.4 * float(normalized[idx]) + 0.5 * bonus + 0.2 * float(match[idx])
        item = docs[idx]
        result.append({
            "index": idx, "id": item.id, "title": item.title, "text": item.text,
            "relevance_score": float(scores[idx]),
            "normalized_relevance": float(normalized[idx]),
            "directed_link_bonus": bonus,
            "query_title_match": float(match[idx]),
            "step_score": step,
        })
        selected.append(idx)
    return result
