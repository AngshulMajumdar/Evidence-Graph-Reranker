# Install and use Evidence Graph Reranker

The released method is a training-free, query-conditioned **subset selector**. It accepts a query, candidate passages, and neural relevance scores, then greedily selects complementary passages. It does **not** perform corpus retrieval or generate answers by itself.

## Requirements

Python 3.10+; NumPy is the only required dependency for using supplied scores. CPU is supported. No environment variables, authentication token, GPU, downloaded model, or external service is needed for the offline API.

From the root of this repository:

```bash
python -m pip install -e .
python -m evidence_graph_reranker --query "Who founded Novus Labs?" --input examples/passages.json --scores "0.95,0.15,0.40,0.10" --top-k 2
```

## Python API — no model download

```python
from evidence_graph_reranker import rank_with_scores

query = "Who founded the company that released the Atlas device?"
passages = [
    {"title": "Atlas device", "text": "The Atlas device was released by Novus Labs."},
    {"title": "Novus Labs", "text": "Novus Labs was founded by Mira Chen."},
    {"title": "Other company", "text": "Unrelated document."}
]
neural_scores = [0.95, 0.15, 0.40] # Your existing scorer's output
selection = rank_with_scores(query, passages, neural_scores, k=2)
print([(p["index"], p["title"]) for p in selection])
```

Scores may come from any real relevance model. The reranker applies per-query min–max normalization before the same fixed 0.4/0.5/0.2 selection rule. Passage dictionaries may contain `title`, `text` (or `body`), and optional `id`. Titles are optional; without them there is no title-link bonus.

## Python API — neural scorers

Install the optional pretrained models interface:

```bash
python -m pip install -e '.[neural]'
```

```python
from evidence_graph_reranker import EvidenceGraphReranker

reranker = EvidenceGraphReranker(backend="cross-encoder", device="cpu")
selected = reranker.rank(query, passages, k=2)
```

Valid backends: `cross-encoder` uses `cross-encoder/ms-marco-MiniLM-L6-v2`; `dense` uses `sentence-transformers/all-MiniLM-L6-v2`. Their weights are downloaded on first use from the original model repositories, subject to those authors' licensing terms. The graph selector itself is unchanged across backends. The first neural download requires internet access.

## Command line

From the repository root after installation:

```bash
# Existing relevance scores (fully offline)
python -m evidence_graph_reranker --input examples/scored_request.json --top-k 3

# Interactive neural scoring (downloads official weights on first run)
python -m evidence_graph_reranker --input examples/passages.json --query "Who founded the company that released the Atlas device?" --backend cross-encoder --top-k 3

# Write JSON result to a file
python -m evidence_graph_reranker --input examples/scored_request.json --top-k 3 --output selected.json
```

If the input is a JSON object, it can contain `query`, `passages`, and `scores`. Otherwise supply a JSON array of passages with `--query` and either `--scores` (comma-separated) or `--backend`.

## Web demo

```bash
python -m pip install -e '.[demo]'
python app.py
```

Open the local URL printed by Gradio. The same root-level `app.py` also works in a Gradio Hugging Face Space. For Spaces, copy the contents of `HUGGINGFACE_SPACE_README.md` to its `README.md` because GitHub's root README is deliberately benchmark-table-focused.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python benchmarks/verify_saved_results.py
```

## Evaluate your own candidates

```bash
python benchmarks/evaluate_jsonl.py --input examples/evaluation.jsonl --k 3 --output evaluation_summary.json
```

Each JSONL row should have `query`, `passages`, `scores`, and `gold_indices` (zero-based indices of all necessary supporting passages). The evaluator compares the single fixed selector to the independent neural-score ranking, and measures **complete supporting evidence coverage** rather than answer generation quality.

## Documented limitations

The string-link graph benefits from explicit titles and cross-references; on untitled fact passages the graph bonus vanishes and selection reduces to ordinary neural relevance ranking. The six benchmark protocols differ, especially in candidate retrieval and gold availability. The observed CPU graph times exclude neural scoring; do not treat them as end-to-end times. See `MODEL_CARD.md` and `results/methodology.md`.

This repository does not bundle third-party neural weights or full benchmark datasets. The MIT license applies to selector source code, not to external weights or corpora.
