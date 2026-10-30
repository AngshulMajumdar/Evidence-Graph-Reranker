# Evidence Graph Reranker

**Evidence recovery @3 (%, higher is better).** One fixed algorithm and identical coefficients across all six benchmarks. Gold evidence is used for evaluation, not graph selection.

| Benchmark | Questions | Dense | MS MARCO | Dense + graph | MS MARCO + graph | Gain (pp) |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 600 | 50.67 | 60.83 | 74.83 | **80.50** | **+19.67** |
| 2WikiMultiHopQA | 600 | 34.17 | 44.50 | 68.33 | **70.17** | **+25.67** |
| HoVer | 300 | 47.33 | 57.00 | 73.33 | **72.67** | **+15.67** |
| MuSiQue | 300 | 16.00 | 19.33 | 25.00 | **27.00** | **+7.67** |
| MultiHop-RAG | 300 | 11.67 | 13.67 | 11.67 | **13.67** | **0.00** |
| QASC | 300 | 5.33 | 13.00 | 5.33 | **13.00** | **0.00** |

**CPU timing (ms/question, measured in the original experiments).** Neural scoring and graph selection are separate stages, not alternative end-to-end runtimes; their measurement conditions vary across benchmarks.

| Benchmark | Cross-encoder scoring | Graph feature extraction / selection |
|---|---:|---:|
| HotpotQA | 272.18 | 0.406 |
| 2WikiMultiHopQA | 254.92 | 0.322 |
| HoVer | 280.38 | 0.370 |
| MuSiQue | 538.67 | 0.662 |
| MultiHop-RAG | 635.04 | 7.570 |
| QASC | 69.12 | 0.340 |

**Reporting scope (at k=3).** The first three Wikipedia-style benchmarks and MuSiQue use benchmark-provided or matched gold-containing candidate pools; MultiHop-RAG and QASC use MiniLM retrieval without gold insertion. HoVer is a restricted matched subset; this metric is complete annotated evidence coverage, **not answer accuracy**.

| Metric | Value |
|---|---:|
| Benchmarks | 6 |
| Questions evaluated in total (different protocols) | 2,400 |
| Improved at k=3 | 4 |
| Tied at k=3 | 2 |
| Regressed at k=3 | 0 |
| Fixed graph coefficients (relevance / link / title) | 0.4 / 0.5 / 0.2 |

**[Installation and Python/CLI/API usage](INSTALL.md) · [GitHub Pages article](docs/index.html) · [Medium-ready article](docs/article.md) · [Detailed methodology](results/methodology.md) · [Hugging Face demo](https://huggingface.co/spaces/Angshul/Evidence-Graph-Reranker)**
