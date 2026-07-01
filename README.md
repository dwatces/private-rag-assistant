# Private Firm Knowledge Assistant

A grounded question-answering assistant over a firm's **own** documents — built
for small professional-services firms (accounting, bookkeeping, legal) that hold
sensitive client data and can't safely paste it into public AI tools.

Staff ask a question in plain English; the assistant answers using **only** the
firm's documents, **cites the source** of every answer, and says *"that isn't in
the firm's documents"* rather than hallucinating. It runs entirely on the firm's
own machine — **no client or firm data leaves the building.**

## Why it's different

- **Private by design.** Documents are embedded and searched locally; nothing is
  sent to a third-party cloud unless the firm chooses to. This is the feature
  regulated firms actually pay for.
- **Grounded + cited.** Answers are constrained to retrieved passages and carry
  citations, so staff can verify them — no confident nonsense.
- **No hallucinated policy.** If the answer isn't in the documents, it says so.

## Example interactions

> **Q: Can I paste a client's tax return into ChatGPT to summarize it?**
>
> No. Firm policy prohibits pasting client information into public, third-party
> cloud AI services like ChatGPT. AI tools are permitted only when client data
> stays on firm-controlled infrastructure — to summarize a client tax return,
> use the internal knowledge assistant, which runs entirely on firm hardware so
> no client data leaves the building **[1]**.
>
> *Sources: 05_data_security_and_privacy.md › Private AI Policy*

> **Q: How do we onboard a new business client, and what documents do we need?**
>
> Five-step intake **[1]**: send the welcome packet + engagement letter; collect
> prior two years of returns, ID, and entity documents; set the client up in the
> practice-management system and portal; assign a lead accountant; schedule a
> kickoff call within five business days. For businesses specifically you need
> the prior-year business return, formation documents, EIN letter, prior-year
> financial statements, and a list of authorized contacts **[3]**.
>
> *Sources: 01_client_onboarding.md › New Client Intake Process; › Required Documents by Client Type*

## How it works

```
documents ──▶ chunk by heading ──▶ embed locally ──▶ cosine retrieval ──▶ grounded answer (cited)
              (corpus/*.md)         (sentence-          (numpy)            (LLM, pluggable backend)
                                     transformers)
```

- **Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` embeddings + numpy
  cosine search. Fully local, no GPU required.
- **Generation backends** (pluggable, see `rag.py`):
  - `claude-cli` — uses an existing Claude subscription via the Claude Code CLI; no API key. *(used for this demo)*
  - `claude-api` — uses the Anthropic API with the client's own key. *(the shipped-product path)*
  - `extractive` — no LLM at all; returns the cited passages directly. Fully offline, $0.
  - A fully-local LLM backend (e.g. a small quantized model) is the drop-in for a true air-gapped deployment.

## Run it

```bash
# command line
python rag.py "What are our standard billing rates?"
python rag.py --interactive

# web UI
python app.py        # then open http://localhost:7860
```

## Notes

The corpus here is a fictional firm's internal handbook (`corpus/`), used so the
demo is self-contained and carries no real client data. In a deployment the firm
points it at their own documents. For a true on-premise install, the generation
step uses a local model so the system has no external dependencies at all.

## License

Released under the [MIT License](LICENSE) — © 2026 Daniel Olliver.
