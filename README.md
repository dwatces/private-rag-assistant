# Private RAG Assistant

A small retrieval-augmented question-answering prototype that answers questions from a folder of your own Markdown or text documents and cites the passages it used.

## What it is

This is a prototype knowledge assistant for a small office, written with an accounting firm's internal handbook in mind. Staff ask a question in plain English and get an answer drawn from the firm's own documents, with numbered citations to the passages it used.

Embedding and search run locally on your machine. Answer generation is pluggable: you can stay fully offline and get the cited passages back, or have an LLM write a short answer from them.

## Features

- Loads every `.md` and `.txt` file in `corpus/` (subfolders are not searched), splits each file on its Markdown headings, then into overlapping windows of up to 150 words (a new window every 120 words), so every passage can be cited by file and section.
- Embeds passages locally with `sentence-transformers/all-MiniLM-L6-v2` and ranks them by cosine similarity with NumPy. No GPU or vector database is needed.
- Retrieves the top 4 passages by default (change it with `--k`).
- Answer backends, chosen with `--backend`:
  - `extractive`: no LLM. Prints the retrieved passages with their citations.
  - `claude-cli`: runs the Claude Code CLI (`claude -p`) with a grounding system prompt, using an existing Claude login rather than an API key.
  - `claude-api`: calls the Anthropic Messages API (needs `ANTHROPIC_API_KEY`).
  - `auto` (the default): `claude-cli` if the `claude` command is on your PATH, otherwise `claude-api` if `ANTHROPIC_API_KEY` is set, otherwise `extractive`.
- The system prompt tells the model to answer only from the numbered passages, cite them as [1], [2], and say the answer is not in the documents rather than guess.
- `--dry-run` prints the exact system and user prompt instead of calling a model.
- A Gradio web page (`app.py`) with a question box, example questions and a sources panel listing each passage's file, section and relevance score.

## What stays on your machine

- Embedding and retrieval are local. The embedding model is downloaded from Hugging Face the first time it runs. After that, your documents are not sent anywhere to be searched.
- With `--backend extractive`, your questions and documents stay on the machine.
- The `claude-cli` and `claude-api` backends send your question and the retrieved passages to Anthropic to write the answer.
- `auto` picks `claude-cli` whenever the `claude` command is installed, so pass `--backend extractive` if you need to stay offline.
- The web UI always uses the `claude-cli` backend.
- There is no local LLM backend in this repository.

## Tech stack

- Python
- sentence-transformers and NumPy for embeddings and search
- Gradio for the web UI
- Optional: the Anthropic Python SDK (`claude-api` backend) or the Claude Code CLI (`claude-cli` backend and the web UI)

## Run it locally

### 1. Install

There is no requirements file. These are the packages the code imports. The web UI needs Gradio 6 or later, because `app.py` passes its theme to `launch()`, which older Gradio versions do not accept. Gradio 6 and current sentence-transformers need Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy sentence-transformers "gradio>=6"
pip install anthropic   # only needed for the claude-api backend
```

### 2. Add your documents

The repository does not include any documents. Create a `corpus/` folder next to `rag.py` and put your `.md` or `.txt` files in it. Markdown headings become the section names shown in citations. Until `corpus/` has files in it, `rag.py` stops with "No documents found".

`corpus/` is not listed in `.gitignore`, so take care not to commit private documents.

### 3. Ask questions from the command line

```bash
python rag.py "What is our client onboarding process?"
python rag.py --interactive
python rag.py --backend extractive "What are our billing rates?"
python rag.py --dry-run "Can I use ChatGPT with client data?"
```

| Option | Default | What it does |
| --- | --- | --- |
| `--backend` | `auto` | `auto`, `extractive`, `claude-cli` or `claude-api` |
| `--model` | `claude-haiku-4-5` | Model used by the Claude backends |
| `--k` | `4` | Number of passages to retrieve |
| `--dry-run` | off | Print the grounded prompt instead of calling a model |
| `--interactive` | off | Keep asking questions until a blank line, `exit` or `quit` |

Each answer is followed by a `Sources:` line listing the file and section of every retrieved passage.

### 4. Or use the web UI

```bash
python app.py
```

Then open http://localhost:7860 (it listens on 127.0.0.1 only). The web UI needs documents in `corpus/`, and the Claude Code CLI installed and signed in, because it always uses the `claude-cli` backend.

The page title, heading and example questions are written for the fictional accounting firm the demo was built around ("Birchwood & Vale, CPAs"). Edit `app.py` to change them.

### Environment variables

- `ANTHROPIC_API_KEY`: used by the `claude-api` backend, and checked by `auto` when choosing a backend.

## Project structure

```
rag.py      Loading, chunking, retrieval, prompt building, backends and the CLI
app.py      Gradio web UI built on rag.py
corpus/     Your documents (not included, create it yourself)
LICENSE     MIT licence
```

## Limitations

- The index is rebuilt in memory each time the script starts. It is not saved to disk.
- It is a prototype: there are no tests, no requirements file and no login on the web UI.

## Licence

Released under the [MIT License](LICENSE). Copyright (c) 2026 Daniel Olliver.

## Author

Daniel Olliver, [danielolliver.com](https://danielolliver.com/)
