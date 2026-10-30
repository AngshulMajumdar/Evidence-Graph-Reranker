"""CLI for the public selector; no model download when supplied scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Sequence
from .core import rank_with_scores


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select complementary evidence using the fixed graph rule.")
    parser.add_argument("--input", required=True, help="JSON list of passages or request with query/passages/scores")
    parser.add_argument("--query", help="Required if not contained in the input JSON object")
    parser.add_argument("--scores", help="Comma-separated relevance scores (optional when JSON contains scores)")
    parser.add_argument("--backend", choices=["cross-encoder", "dense"], default="cross-encoder",
                        help="Used only when neither --scores nor input scores are supplied")
    parser.add_argument("--device", default="cpu", help="Torch device for optional neural scoring")
    parser.add_argument("--top-k", type=int, default=3, help="Number of selected passages")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args(argv)
    with open(args.input, encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, list):
        docs, query, scores = loaded, args.query, None
    elif isinstance(loaded, dict):
        docs, query, scores = loaded.get("passages"), args.query or loaded.get("query"), loaded.get("scores")
    else:
        parser.error("Input must be a JSON list or an object with passages")
    if not isinstance(docs, list) or not docs:
        parser.error("Input must contain a nonempty list of passages")
    if not isinstance(query, str) or not query.strip():
        parser.error("Provide a nonempty --query or JSON 'query'")
    if args.scores is not None:
        try:
            scores = [float(v.strip()) for v in args.scores.split(",")]
        except ValueError:
            parser.error("--scores must be comma-separated numeric values")
    if scores is None:
        from .neural import EvidenceGraphReranker
        selected = EvidenceGraphReranker(backend=args.backend,device=args.device).rank(query,docs,k=args.top_k)
    else:
        selected = rank_with_scores(query, docs, scores, k=args.top_k)
    result = json.dumps({"query": query, "selected": selected}, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
