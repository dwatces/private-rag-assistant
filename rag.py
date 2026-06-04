"""
Private Firm Knowledge Assistant — local RAG over a firm's own documents.

Documents are embedded locally with a sentence-transformers model and retrieved
with cosine similarity, so the firm's content never leaves the machine. Answer
generation is pluggable:

  * extractive   — stitches the most relevant passages with citations; no LLM
                   at all (fully offline, $0).
  * claude-cli   — generates a grounded answer via the Claude Code CLI
                   (`claude -p`), using an existing Claude subscription. No API
                   key needed. Great for the demo / your own use.
  * claude-api   — same, via the Anthropic API (needs ANTHROPIC_API_KEY). This
                   is the path a shipped product would use with a client's key.

Usage:
    python rag.py "What is our client onboarding process?"
    python rag.py --interactive
    python rag.py --backend extractive "What are our billing rates?"
    python rag.py --dry-run "Can I use ChatGPT with client data?"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "claude-haiku-4-5"   # fast + cheap; bump to claude-sonnet-4-6 for polish
TOP_K = 4
WORDS_PER_CHUNK = 120
CHUNK_OVERLAP = 30

SYSTEM_PROMPT = (
    "You are the internal knowledge assistant for an accounting firm. Answer "
    "the staff member's question using ONLY the numbered context passages "
    "provided. Cite the passages you use like [1], [2]. Be concise and "
    "practical. If the answer is not contained in the context, say you don't "
    "have that in the firm's documents — never guess."
)


@dataclass
class Chunk:
    doc: str       # source filename, e.g. "01_client_onboarding.md"
    section: str   # nearest markdown heading, used in citations
    text: str


def load_chunks(corpus_dir: str) -> list[Chunk]:
    """Read every .md/.txt file, split on markdown headings, then into
    overlapping word windows so each chunk is a citable, focused passage."""
    chunks: list[Chunk] = []
    paths = sorted(glob.glob(os.path.join(corpus_dir, "*.md")) +
                   glob.glob(os.path.join(corpus_dir, "*.txt")))
    for path in paths:
        doc = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        # capturing-group split -> [preamble, head, body, head, body, ...]
        parts = re.split(r"(?m)^(#{1,6}\s+.*)$", raw)

        def emit(section: str, body: str) -> None:
            words = body.split()
            if not words:
                return
            span = WORDS_PER_CHUNK + CHUNK_OVERLAP
            for i in range(0, len(words), WORDS_PER_CHUNK):
                window = " ".join(words[i:i + span]).strip()
                if window:
                    chunks.append(Chunk(doc=doc, section=section, text=window))

        emit(doc, parts[0])  # text before the first heading
        for i in range(1, len(parts), 2):
            section = parts[i].lstrip("#").strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            emit(section, body)
    return chunks


class Retriever:
    """Local semantic search: embed once, cosine-search per query."""

    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.mat = self.model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        scores = self.mat @ q                      # cosine (vectors normalized)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


# --------------------------- prompt + backends ----------------------------

def build_prompt(query: str, hits: list[tuple[Chunk, float]]) -> tuple[str, str]:
    """Return (system, user) for a grounded, citable answer."""
    context = "\n\n".join(
        f"[{i + 1}] ({c.doc} › {c.section})\n{c.text}"
        for i, (c, _) in enumerate(hits)
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return SYSTEM_PROMPT, user


def answer_extractive(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """No-LLM fallback: present the grounding passages directly with citations."""
    blocks = [f"[{i + 1}] ({c.doc} › {c.section})\n    {c.text}"
              for i, (c, _) in enumerate(hits)]
    return ("Most relevant passages from the firm's documents:\n\n"
            + "\n\n".join(blocks))


def answer_claude_cli(query: str, hits: list[tuple[Chunk, float]],
                      model: str = DEFAULT_LLM) -> str:
    """Grounded generation via the Claude Code CLI, on an existing subscription
    (no API key). Passes the prompt as argv (no shell) so document text with
    quotes/newlines is safe."""
    system, user = build_prompt(query, hits)
    cmd = ["claude", "-p", "--system-prompt", system, "--model", model, user]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:400]}")
    return proc.stdout.strip()


def answer_claude_api(query: str, hits: list[tuple[Chunk, float]],
                      model: str = DEFAULT_LLM) -> str:
    """Grounded generation via the Anthropic API (requires ANTHROPIC_API_KEY).
    This is the path a shipped product uses with the client's own key."""
    import anthropic
    system, user = build_prompt(query, hits)
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=600, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def pick_backend(name: str) -> str:
    if name != "auto":
        return name
    if shutil.which("claude"):
        return "claude-cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-api"
    return "extractive"


# --------------------------------- cli ------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Private firm knowledge assistant (local RAG).")
    ap.add_argument("query", nargs="*", help="question to ask")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "extractive", "claude-cli", "claude-api"])
    ap.add_argument("--model", default=DEFAULT_LLM, help="LLM for generation")
    ap.add_argument("--k", type=int, default=TOP_K, help="passages to retrieve")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grounded prompt instead of calling the LLM")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    print("Loading firm documents and building local index...", file=sys.stderr)
    chunks = load_chunks(CORPUS_DIR)
    if not chunks:
        sys.exit(f"No documents found in {CORPUS_DIR}")
    retriever = Retriever(chunks)
    ndocs = len({c.doc for c in chunks})
    print(f"Indexed {len(chunks)} passages from {ndocs} documents.", file=sys.stderr)

    backend = pick_backend(args.backend)
    print(f"[backend: {backend}{' (dry-run)' if args.dry_run else ''}]\n", file=sys.stderr)

    backends = {
        "extractive": answer_extractive,
        "claude-cli": answer_claude_cli,
        "claude-api": answer_claude_api,
    }

    def run(question: str) -> None:
        hits = retriever.search(question, k=args.k)
        if args.dry_run:
            system, user = build_prompt(question, hits)
            print(f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}")
        elif backend == "extractive":
            print(answer_extractive(question, hits))
        else:
            print(backends[backend](question, hits, args.model))
        cites = ", ".join(f"{c.doc}›{c.section}" for c, _ in hits)
        print(f"\nSources: {cites}")

    if args.interactive:
        print("Ask the firm assistant a question (blank line or 'exit' to quit).")
        while True:
            try:
                q = input("\nAsk> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"quit", "exit"}:
                break
            run(q)
    else:
        run(" ".join(args.query) or "What is our client onboarding process?")


if __name__ == "__main__":
    main()
