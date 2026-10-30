"""Query-conditioned Evidence-Graph Reranker (one fixed algorithm)."""
from .core import Passage, graph_features, minmax_relevance, select_with_features, rank_with_scores
from .neural import EvidenceGraphReranker, DENSE_MODEL, CROSS_ENCODER_MODEL
__all__ = ["Passage", "graph_features", "minmax_relevance", "select_with_features",
           "rank_with_scores", "EvidenceGraphReranker", "DENSE_MODEL", "CROSS_ENCODER_MODEL"]
__version__ = "0.1.0"
