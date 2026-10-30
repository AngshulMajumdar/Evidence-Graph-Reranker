# Benchmarks and reproducibility

- `evaluate_jsonl.py` is a portable external-dataset evaluator. It takes per-question candidate pools, **neural relevance scores**, and gold indices. It does not retrieve or generate answers, and it never uses gold to rank.
- `verify_saved_results.py` audits the saved 600-question HotpotQA and 2Wiki result files against the published aggregate CSV for four budgets and four ranking options.
- `results/six_dataset_results.csv`, `results/six_dataset_results.json`, and `results/methodology.md` record the six different evaluation protocols, caveats, and CPU stage timings. All use the same selector coefficients.
- The `results/unified_*_per_question.json` files include the original two's selected orders and gold-hit indicators; they omit passage texts and neural scores, so **cannot** be used to regenerate rankings without the original candidate pools and embeddings.
- The original benchmark datasets and third-party pretrained weights are not included. Full neural forward-pass reproduction requires acquiring those inputs and running the published model identifiers. This release does not claim full reproduction with only the included files.

```bash
python benchmarks/verify_saved_results.py
python benchmarks/evaluate_jsonl.py --input examples/evaluation.jsonl --k 2
```
