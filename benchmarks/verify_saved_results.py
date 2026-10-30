"""Verify saved per-question results match the published aggregate tables.

Does not rerun neural inference or claim to remeasure CPU latency.
"""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
ROWS = list(csv.DictReader((ROOT/'results/six_dataset_results.csv').open(encoding='utf-8')))


def verify():
    for dataset, stub in [('HotpotQA','hotpot'),('2WikiMultiHopQA','wiki')]:
        cases=json.loads((ROOT/f'results/unified_{stub}_per_question.json').read_text(encoding='utf-8'))
        assert len(cases)==600, f'{dataset}: unexpected number of cases'
        for k in [2,3,5,8]:
            row=next(r for r in ROWS if r['dataset']==dataset and int(r['budget_k'])==k)
            for method,column in [('dense_cosine','dense_cosine_pct'),('crossencoder','crossencoder_pct'),
                                  ('dense_graph','dense_plus_graph_pct'),('crossencoder_graph','crossencoder_plus_graph_pct')]:
                observed=round(100*sum(q['hits'][method][str(k)] for q in cases)/len(cases),2)
                expected=float(row[column])
                assert abs(observed-expected)<0.011, (dataset,k,method,observed,expected)
    print('PASS: HotpotQA and 2Wiki 600-question outcomes match all @2/@3/@5/@8 aggregates.')
    print('Other four datasets are distributed as aggregate summaries; not recomputed here.')

if __name__ == '__main__': verify()
