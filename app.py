"""Hugging Face Space: live CPU demo for the exact fixed evidence-graph reranker."""
import json
import gradio as gr
from evidence_graph_reranker import EvidenceGraphReranker

_RERANKERS = {}
EXAMPLE = [
    {"title": "Atlas device", "text": "The Atlas device was launched by Novus Labs in 2025."},
    {"title": "Novus Labs", "text": "Novus Labs was founded by Mira Chen in 2018."},
    {"title": "Orion spacecraft", "text": "The Orion spacecraft is designed for space missions."},
    {"title": "Electronics", "text": "Electronics can be manufactured in many countries."},
]


def demo(query, passages_json, budget, backend):
    try:
        docs = json.loads(passages_json)
        if not isinstance(docs, list) or not 1 <= len(docs) <= 100:
            raise ValueError("Enter a JSON list containing 1–100 passages.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if backend not in _RERANKERS:
            _RERANKERS[backend] = EvidenceGraphReranker(backend=backend, device="cpu", batch_size=16)
        ranked = _RERANKERS[backend].rank(query, docs, k=int(budget))
        table = [
            [i + 1, r["index"], r["title"] or "(untitled)",
             round(r["relevance_score"], 4), round(r["step_score"], 4),
             round(r["directed_link_bonus"], 2)]
            for i, r in enumerate(ranked)
        ]
        return table, json.dumps(ranked, indent=2, ensure_ascii=False)
    except Exception as exc:
        return [], json.dumps({"error": str(exc)})

with gr.Blocks(title="Evidence-Graph Reranker") as app:
    gr.Markdown("# Evidence-Graph Reranker\nTraining-free, query-conditioned **joint evidence selection** after neural relevance scoring.\n\nA passage's usefulness can rise when it completes a link from a previously selected passage. This demo uses **one fixed graph rule** for every dataset. Model weights are fetched from their original Hugging Face repositories.")
    query = gr.Textbox(label="Query", value="Who founded the company that released the Atlas device?")
    passages = gr.Code(label="Candidate passages (JSON array; optional title)", language="json",
                       value=json.dumps(EXAMPLE, indent=2))
    with gr.Row():
        budget = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Passage budget (k)")
        backend = gr.Dropdown(choices=["cross-encoder", "dense"], value="cross-encoder",
                              label="Pretrained neural relevance model")
    go = gr.Button("Select complementary evidence", variant="primary")
    table = gr.Dataframe(headers=["Rank", "Candidate index", "Title", "Neural score", "Selection score", "Link bonus"],
                         datatype=["number", "number", "str", "number", "number", "number"],
                         label="Selected passages", interactive=False)
    output = gr.Code(label="Full selection JSON", language="json")
    go.click(demo, inputs=[query, passages, budget, backend], outputs=[table, output])
    gr.Markdown("**Interpretation:** `selection_score` is a step-dependent heuristic, not a probability. Gold labels are never used by the selector. Reported benchmarks measure complete evidence coverage, *not* LLM-generated answer accuracy. See `MODEL_CARD.md` for protocols and caveats.")

if __name__ == "__main__":
    app.launch()
