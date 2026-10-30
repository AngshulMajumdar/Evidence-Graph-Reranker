import json
from pathlib import Path
import numpy as np
import pytest
from evidence_graph_reranker import (Passage, graph_features, select_with_features,
                                     rank_with_scores, EvidenceGraphReranker)
from evidence_graph_reranker.cli import main


def test_no_links_no_titles_preserves_score_order():
    docs=['abc','xyz','pqr']
    out=rank_with_scores('anything',docs,[0.1,0.9,0.5],k=3)
    assert [r['index'] for r in out]==[1,2,0]


def test_chain_complementarity():
    docs=[Passage('Atlas device','Atlas device was released by Novus Labs'),
          Passage('Novus Labs','Novus Labs was founded by Mira Chen'),
          Passage('Other company','This is irrelevant')]
    q='Who founded the company that released the Atlas device?'
    link,match=graph_features(q,docs)
    assert link[0,1]==1.0
    assert select_with_features([0.95,0.15,0.4],link,match,k=2)==[0,1]


def test_no_models_when_scores_supplied():
    model=EvidenceGraphReranker()
    out=model.rank('A',['x','y'],scores=[1.0,.3],k=1)
    assert [r['index'] for r in out]==[0] and model._model is None


def test_deterministic_ties_and_limits():
    assert [r['index'] for r in rank_with_scores('a',['x','y','z'],[1,1,1],k=5)] == [0,1,2]
    with pytest.raises(ValueError): rank_with_scores('q',['a','b'],[1],k=2)
    with pytest.raises(ValueError): rank_with_scores('q',['a'],[float('nan')],k=1)
    with pytest.raises(ValueError): rank_with_scores('q',['a'],[1],k=0)


def test_cli_offline_json(capsys):
    inp=Path(__file__).resolve().parents[1]/'examples/scored_request.json'
    assert main(['--input',str(inp),'--top-k','2'])==0
    obj=json.loads(capsys.readouterr().out)
    assert [p['index'] for p in obj['selected']]==[0,1]


def test_graph_matches_simple_reference():
    q='Who founded Novus Labs?'
    docs=[Passage('Atlas','The Atlas was made by Novus Labs'),
          Passage('Novus Labs','Novus Labs founded by Mira Chen'),
          Passage('Mira Chen','A scientist'), Passage('X','irrelevant')]
    link,match=graph_features(q,docs)
    assert link[0,1] == 1
    assert match[1] == 1
    rng=np.random.default_rng(91)
    for _ in range(80):
        scores=rng.normal(size=4)
        observed=select_with_features(scores,link,match,k=3)
        normalized=(scores-scores.min())/(scores.max()-scores.min())
        selected=[]
        for t in range(3):
            remain=[j for j in range(4) if j not in selected]
            j=max(remain,key=lambda j:(.4*normalized[j]+.5*max([link[i,j] for i in selected],default=0)+.2*match[j],-j))
            selected.append(j)
        assert observed==selected
