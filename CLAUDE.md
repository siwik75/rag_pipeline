# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A lossless RAG pipeline over trading-strategy PDFs, plus a live technical-analysis tool that queries the same
knowledge base. Three scripts, no framework: `ingest.py` (PDF → text → chunks → embeddings → ChromaDB),
`query.py` (retrieval + Claude-generated answer with citations), and `signal_check.py` (live OHLCV indicators →
interprets them → asks the RAG what the books say about that scenario). All behavior is centralized in
`config.py`. User-facing strings, CLI help, and the README are in Italian; code identifiers are in English.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest PDFs (incremental — skips files already indexed by SHA-256 hash)
python ingest.py
python ingest.py --force        # re-embed everything, e.g. after changing EMBEDDING_MODEL

# Query the knowledge base
python query.py "your question"
python query.py "position sizing" --k 8 --no-llm    # retrieval only, no Claude call

# Live technical analysis + RAG
python signal_check.py BTC
python signal_check.py BTC ETH SOL --tf 1d
python signal_check.py BTC --no-llm                  # dashboard + retrieved passages only
python signal_check.py BTC --exchange kraken --quote USD
```

There is no test suite, linter, or build step in this repo.

Claude calls require `ANTHROPIC_API_KEY`, either in `.env` (copy from `.env.example`, auto-loaded via
python-dotenv) or exported in the shell. Without it, `query.py` and `signal_check.py` silently degrade to
retrieval-only output instead of erroring — preserve this fallback when touching either script.

## Architecture

**`config.py`** is the single source of truth for paths, the embedding model, chunking sizes, and the Claude
model name. Read it before changing behavior in any other script — there are no hardcoded constants duplicated
elsewhere.

- `PDF_DIR = PROJECT_ROOT.parent` — source PDFs live **one directory above** `rag_pipeline/`, not inside it.
  `ingest.py` globs `*.pdf` there.
- Embeddings are local via `fastembed`/ONNX (no torch), using a multilingual model
  (`paraphrase-multilingual-MiniLM-L12-v2` by default) so Italian queries can match English source text.
  Swapping to an E5 model (e.g. `intfloat/multilingual-e5-large`) requires re-running `ingest.py --force`
  because E5 models need `passage:`/`query:` prefixes — `config.py` derives `PASSAGE_PREFIX`/`QUERY_PREFIX`
  automatically from whether `"e5"` appears in `EMBEDDING_MODEL`, and both `ingest.py` and `query.py` must keep
  using matching prefixes for a given embedding space.

**`ingest.py`** pipeline per PDF: extract text page-by-page with PyMuPDF, fall back to `pytesseract` OCR per
page when extracted text is under `MIN_CHARS_PER_PAGE` (OCR is optional — code checks `OCR_AVAILABLE` and
degrades gracefully if `pytesseract`/`Pillow` aren't installed), append any tables as Markdown. The full
per-page text is written to `extracted/<name>.md` as a lossless archive — this is separate from and unrelated
to what gets embedded. Text is then chunked (paragraph-aware, `CHUNK_SIZE`/`CHUNK_OVERLAP` from `config.py`,
tracking `page_start`/`page_end` per chunk for citations) and upserted into the ChromaDB collection
`trading_knowledge` in `vector_store/`. Idempotency is done via SHA-256 file hashes stored in
`ingest_manifest.json`: unchanged files are skipped entirely; changed files have their old chunks deleted
(`collection.delete(where={"source": pdf.name})`) before re-adding.

**`query.py`** exposes `retrieve(question, k)` and `format_hits(hits)` as the reusable retrieval API — both are
imported directly by `signal_check.py`, so changes to the hit dict shape (`text`/`meta`/`score`) or citation
format ripple into that script too. `ask_claude` enforces citation-only answers via the system prompt (no
answers beyond the retrieved context).

**`signal_check.py`** fetches OHLCV via `ccxt` (public endpoints, no exchange API key needed), computes ~50
indicators with `ta` (momentum, trend, volatility, volume — see `compute_indicators`), reduces them to plain-
language states in `interpret()`, and feeds those states as a retrieval query into `query.retrieve()` before
asking Claude how the source books would frame that scenario. It is not standalone — it depends on `query.py`
for retrieval, and on `vector_store/` already being populated via `ingest.py`.

## Notes for future changes

- Any change to `EMBEDDING_MODEL` invalidates the existing vector store (different embedding space) and
  requires `ingest.py --force`.
- The retrieval → LLM fallback (`--no-llm` or missing `ANTHROPIC_API_KEY`) is implemented independently in both
  `query.py` and `signal_check.py`; keep both in sync if the fallback UX changes.
- Both `signal_check.py` and `query.ask_claude` treat this as an informational tool, not financial advice, and
  say so in prompts/output — preserve that framing in any new user-facing text.
