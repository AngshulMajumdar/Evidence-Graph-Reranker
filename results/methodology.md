# Six-dataset benchmark methodology and measured findings

## Exactly one selector

All six results here were scored with the same implementation of `graph_arrays` and `graph_select` from the four-dataset neural-only experiments. Public `evidence_graph_reranker` mirrors that code. There is no per-dataset preset. At inference, query and candidate text generate title/body edges and query/title matches, and every remaining item is scored by `0.4 × minmax(neural_relevance) + 0.5 × selected-to-candidate title reference + 0.2 × query/title match`. The actual neural relevance scorer is either all-MiniLM-L6-v2 cosine or pretrained MS MARCO MiniLM-L6-v2 query–passage logits. The algorithm never reads gold labels.

The **relevance backend** is a separate choice, but the selector is identical. Note that the fixed coefficients originated during earlier methodological development; they were not newly tuned on the held-out samples in the unified rerun.

## Complete evidence coverage, percentages

At budget k=3 (selected **passages**, not word/token count):

| Dataset | n | Dense cosine | MS MARCO | Dense + graph | MS MARCO + graph | Gain vs MS MARCO |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 600 | 50.67 | 60.83 | 74.83 | **80.50** | +19.67 pp |
| 2WikiMultiHopQA | 600 | 34.17 | 44.50 | 68.33 | **70.17** | +25.67 pp |
| MuSiQue | 300 | 16.00 | 19.33 | 25.00 | **27.00** | +7.67 pp |
| HoVer | 300 | 47.33 | 57.00 | 73.33 | **72.67** | +15.67 pp |
| MultiHop-RAG | 300 | 11.67 | 13.67 | 11.67 | **13.67** | +0.00 pp |
| QASC | 300 | 5.33 | 13.00 | 5.33 | **13.00** | +0.00 pp |

Results for k=2,3,5,8 appear in `six_dataset_results.csv`; per-question outcomes for the original two are included.

## CPU timing: separate pipeline stages

| Dataset | Crossencoder scorer (ms/q) | Graph feature extraction + selection (ms/q) |
|---|---:|---:|
| HotpotQA | 272.18 | 0.406 |
| 2WikiMultiHopQA | 254.92 | 0.322 |
| MuSiQue | 538.67 | 0.662 |
| HoVer | 280.38 | 0.370 |
| MultiHop-RAG | 635.04 | 7.570 |
| QASC | 69.12 | 0.340 |

**Timing caveat:** the first two graph timings count construction plus four rankings; the other datasets' logs count graph operations within their evaluation loop. CPU hardware and batching are from the original benchmark runs, not normalized across experiments. Neural scorer timings exclude model startup; graph-only timings exclude neural scorer inference. It would be inaccurate to compare 0.4ms to the crossencoder's ~250ms as if the graph pipeline did not require neural relevance scores. The dense+graph pipeline is often much lighter because passage embeddings can be cached, but end-to-end full-pipeline timing requires matched test conditions.

## Original two, unified-rerun integrity

- HotpotQA: 600 held-out queries from 800 seeded HotpotQA distractor examples, first 200 excluded; original pool is 10 passages. Graph+crossencoder CEC@3 80.50%; crossencoder alone 60.83%. Paired 95% bootstrap CI for the +19.67 pp gain [16.17, 23.33].
- 2WikiMultiHopQA: 600 held-out queries from 800 seeded 2Wiki validation examples, first 200 excluded; native candidate pool is 10 passages. Graph+crossencoder CEC@3 70.17%; crossencoder alone 44.50%. Paired 95% bootstrap CI for +25.67 pp [22.17, 29.33].
- The **graph selection and evaluation were rerun** using exact 0.4/0.5/0.2 rules, with original MiniLM query/document vectors and pretrained MS MARCO query/passage logits loaded from previous completed CPU neural inference. We did not rerun the 12,000 neural pairs. The production package's ranking returned the exact same 1,200 top-8 orders as the experimental implementation in an explicit equality check.

## Four additional datasets

- **MuSiQue:** 300 seeded examples, ~20 native candidates per question; complete gold evidence present in pool; question-hop counts vary, so @2 is not always achievable. Frozen pretrained crossencoder logits reused from original neural experiment; dense embeddings recomputed for corrected run.
- **HoVer:** 300 examples selected from **908** claims whose entire annotated evidence set could be matched to uploaded HotpotQA distractor passages. This is a restricted, gold-available conditional pool, not the full 4,000-claim validation benchmark.
- **MultiHop-RAG:** 300 questions, MiniLM retrieval top20 from all 609 uploaded articles; no gold insertion. Only 51.67% have all required sources in top20. Graph boost at 3 passages is 0; at 5 it is +3pp. Exact title links were rare (~4%).
- **QASC:** 300 questions, MiniLM top20 from 6,878 distinct annotated fact sentences (not QASC's full external background corpus), no gold insertion. 31.33% of top20 pools contain both gold facts. Sentences have no titles, so this particular graph rule reduces mathematically to independent neural ranking, causing zero gain.

## Interpretation and limitations

- The metric is *complete annotated supporting-document/fact recovery at k*, not answer accuracy, grounded generation, factual precision, or token-budget-normalized retrieval.
- Gold is used to evaluate, not to score. In HoVer, and in the separate unpublished QASC controlled pool, gold availability matters in candidate pool construction.
- The graph exploits explicit article title references; results on Wikipedia-derived candidate pools may overstate transfer to unstructured corpora.
- This greedy rule has **no proved submodular approximation guarantee**.
- Benchmarks have different candidate pools, n, and retrieval difficulty. Do not pool their percentages as if they were a common IID test set.

## Reproduction pointers

Run `pytest -q` to check the public graph selector. The `reproduce/` directory contains the original benchmark code with the same directory conventions as the experiment (`/mnt/data`); it requires the original datasets and user's downloaded pretrained weights separately. To regenerate benchmark neural scores, use the saved original scripts; to repeat the **single-selector** rerun from existing cached scores, invoke `reproduce/unified_original_two_rerun.py` in that original `/mnt/data` layout. The file snapshots here show the measured outputs; the benchmark source files are not silently included or redownloaded.
