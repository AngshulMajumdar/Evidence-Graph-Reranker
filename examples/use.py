"""Run: python examples/use.py  (network needed for first model download)."""
import json
from evidence_graph_reranker import EvidenceGraphReranker, rank_with_scores

query = "Who founded the company that released the Atlas device?"
with open("examples/passages.json", encoding="utf-8") as f:
    passages = json.load(f)

# Full pretrained MS MARCO -> graph reranking
reranker = EvidenceGraphReranker(backend="cross-encoder", device="cpu")
for passage in reranker.rank(query, passages, k=3):
    print(passage["index"], passage["title"], round(passage["step_score"], 4))

# If you already have neural relevance scores, the selector needs no checkpoint:
print([r["index"] for r in rank_with_scores(query, passages, [.91, .45, .20, .15], k=3)])
