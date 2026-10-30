---
language:
- en
license: mit
tags:
- rag
- reranking
- evidence-subset-selection
- multi-hop-reasoning
pipeline_tag: text-ranking
---

# Evidence-Graph Reranker: algorithm card

**Type:** training-free joint selection algorithm with external pretrained neural relevance scorers. **Not** a standalone trained neural checkpoint or language generator.

## Formula

`score(candidate | already_selected, query) = 0.4 × minmax(relevance) + 0.5 × directed_title_link + 0.2 × query_title_match`

**One rule for all six datasets.** No dataset-specific presets or facet objectives. Scores are recomputed greedily after selection; selecting an informative first passage can make a complementary follow-up passage more attractive.

## Neural models (third-party weights)

- `cross-encoder/ms-marco-MiniLM-L6-v2`: pretrained relevance logits for query/passage pairs, 256 WordPiece cap used in experiments.
- `sentence-transformers/all-MiniLM-L6-v2`: cosine similarity over pretrained normalized embeddings, 128 WordPiece cap used in experiments.

The code pulls these third-party weights separately from Hugging Face. The package's MIT license covers our selector code, not external pretrained weights.

## Evaluation

Measured complete supporting evidence coverage at k=3 passages:

| Benchmark | Cross-encoder | Cross-encoder + fixed graph | Gain |
|---|---:|---:|---:|
| HotpotQA (600) | 60.83% | 80.50% | +19.67 pp |
| 2WikiMultiHopQA (600) | 44.50% | 70.17% | +25.67 pp |
| MuSiQue (300) | 19.33% | 27.00% | +7.67 pp |
| HoVer (300) | 57.00% | 72.67% | +15.67 pp |
| MultiHop-RAG (300) | 13.67% | 13.67% | +0.00 pp |
| QASC (300) | 13.00% | 13.00% | +0.00 pp |


Six datasets, four improvements and two ties at k=3. These are not the same candidate-pool protocol. HotpotQA and 2Wiki use previously computed neural model outputs and newly rerun unified graph selection, not newly recomputed forward passes. For k=2,5,8, exact timings, candidate availability, bootstrap intervals and limitations, see `results/methodology.md` and `results/six_dataset_results.csv`.

## Known limitations

- Direct title/body string linking works best when meaningful titles and article cross-references exist. It cannot reliably model nonlexical semantic connections without additional features, which are *not part of this released single algorithm*.
- On QASC's untitled sentences, the graph features vanish and the method becomes ordinary neural ranking. MultiHop-RAG at k=3 is also tied with the cross-encoder.
- Gold-containing candidate pools on some benchmarks produce an easier, conditional reranking task. The method does not repair missing supporting evidence in the initial retrieval stage.
- **No LLM answer-quality or hallucination-reduction metric was computed.** Avoid equating recovered evidence with successful RAG generation.
- Graph's greedy heuristic does not carry a proven submodular approximation ratio.

## Reuse

Install the package and run the Gradio demo. The graph-only API, `rank_with_scores`, does not download neural checkpoints. Cite the original pretrained models if using neural scoring. Verify the terms of all third-party model checkpoints and datasets.
