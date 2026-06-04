"""Web UI for the private firm knowledge assistant (demo).

    python app.py
        then open http://localhost:7860

Everything runs locally: retrieval is on-machine, and generation uses the
Claude Code subscription via the CLI backend. No client data leaves the box.
"""
import gradio as gr

import rag

print("Building local index...")
_chunks = rag.load_chunks(rag.CORPUS_DIR)
_retriever = rag.Retriever(_chunks)
print(f"Indexed {len(_chunks)} passages from "
      f"{len({c.doc for c in _chunks})} documents.")

EXAMPLES = [
    "Can I paste a client's tax return into ChatGPT to summarize it?",
    "How do we onboard a new business client and what documents do we need?",
    "What are our standard hourly billing rates and when are invoices due?",
    "How much PTO do I get and how do I request it during tax season?",
    "When does a tax return get put on extension?",
]


def respond(question: str):
    question = (question or "").strip()
    if not question:
        return "Ask a question about firm policy or procedures.", ""
    hits = _retriever.search(question, k=rag.TOP_K)
    answer = rag.answer_claude_cli(question, hits)
    sources = "\n".join(
        f"- **{c.doc}** › {c.section}  _(relevance {s:.2f})_"
        for c, s in hits
    )
    return answer, sources


with gr.Blocks(title="Firm Knowledge Assistant") as demo:
    gr.Markdown(
        "# Birchwood &amp; Vale, CPAs — Firm Knowledge Assistant\n"
        "Ask about firm policies and procedures in plain English. Every answer "
        "is grounded **only** in the firm's own documents and shows its sources, "
        "so staff can verify it. If something isn't in the documents, the "
        "assistant says so rather than guessing.\n\n"
        "🔒 Runs entirely on firm infrastructure — **no client data leaves the building.**"
    )
    with gr.Row():
        question = gr.Textbox(
            label="Your question", scale=4,
            placeholder="e.g. What are our standard billing rates?",
        )
        ask = gr.Button("Ask", variant="primary", scale=1)
    answer_md = gr.Markdown(label="Answer")
    with gr.Accordion("Sources (retrieved passages)", open=True):
        sources_md = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=question)

    ask.click(respond, inputs=question, outputs=[answer_md, sources_md])
    question.submit(respond, inputs=question, outputs=[answer_md, sources_md])


if __name__ == "__main__":
    # share=True gives a temporary public link you can send a prospect.
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
