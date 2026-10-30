"""Evaluation of complete gold-evidence coverage on a user-supplied JSONL."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from statistics import mean

from evidence_graph_reranker import rank_with_scores


def evaluate(rows, k):
    if k < 1: raise ValueError("k must be positive")
    complete_graph = []
    complete_base = []
    for line_number, row in enumerate(rows, start=1):
        docs = row["passages"]
        scores = row["scores"]
        gold = set(row["gold_indices"])
        if not gold or any(not isinstance(i,int) or i < 0 or i >= len(docs) for i in gold):
            raise ValueError(f"Invalid gold_indices at record {line_number}")
        graph = [r["index"] for r in rank_with_scores(row["query"],docs,scores,k)]
        # Stable tie-break is the same as the selector: earlier candidate index first.
        base = sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:k]
        complete_graph.append(int(gold.issubset(graph)))
        complete_base.append(int(gold.issubset(base)))
    if not complete_graph:
        raise ValueError("No examples in input")
    return {"questions":len(complete_graph),"k":k,
            "complete_evidence_coverage_graph_pct":100*mean(complete_graph),
            "complete_evidence_coverage_independent_pct":100*mean(complete_base),
            "gain_percentage_points":100*(mean(complete_graph)-mean(complete_base))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--k",type=int,default=3)
    parser.add_argument("--output")
    args=parser.parse_args()
    with open(args.input,encoding="utf-8") as source:
        rows=[json.loads(line) for line in source if line.strip()]
    summary=evaluate(rows,args.k)
    out=json.dumps(summary,indent=2)
    if args.output: Path(args.output).write_text(out+'\n',encoding='utf-8')
    else: print(out)

if __name__=='__main__': main()
