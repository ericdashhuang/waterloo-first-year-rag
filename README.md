# Waterloo First-Year RAG

A retrieval-augmented generation (RAG) system that answers questions about starting first year at the University of Waterloo, including co-op, housing, orientation, and campus wellness.
Built as a learning project, so every step of the pipeline is written in plain Python rather than hidden behind a framework.

## Why this exists

A plain LLM either doesn't know anything about a specific school's policies, or it guesses and gets details wrong.
This project fixes that by retrieving real content from uwaterloo.ca before generating an answer, so responses are grounded in actual source pages instead of the model's memory.

## How it works

1. **Ingest** (`data/ingest.py`) — fetches ~70 public uwaterloo.ca pages (co-op, housing, orientation, international students, campus wellness, registrar) and strips them down to clean article text.
2. **Chunk** (`rag/chunking.py`) — splits each page into ~900-character overlapping chunks along paragraph boundaries, so a chunk rarely cuts a sentence in half.
3. **Embed + index** (`rag/build_index.py`) — converts every chunk into a vector using a local embedding model (`all-MiniLM-L6-v2`, bundled with Chroma, runs on CPU, no API key needed) and stores it in a persistent [Chroma](https://www.trychroma.com/) vector database.
4. **Retrieve + generate** (`rag/query.py`) — embeds the question the same way, finds the most similar chunks by vector search, and passes them to an LLM as context so it can answer from real source material and cite where the answer came from.

No LangChain, no LlamaIndex — every step above is under 100 lines of plain Python, on purpose.
The goal was to actually understand what a RAG pipeline does, not to call a framework method and trust that it works.

## Project structure

```
data/
  seed_urls.txt        curated list of source pages
  discover_urls.py      one-off crawler used to find candidate pages
  ingest.py              fetches + cleans pages into data/raw/
  raw/                    cleaned page text (checked into git)
rag/
  chunking.py            splits page text into overlapping chunks
  build_index.py         embeds chunks and builds the Chroma index
  query.py                retrieval + generation CLI
  export_chunks.py       exports chunk text (no vectors) for the web demo
docs/                    static client-side demo, served by GitHub Pages (see below)
```

## Running it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch and clean the source pages (already included in data/raw/, re-run to refresh)
python data/ingest.py

# 2. Build the vector index (~30 seconds)
python rag/build_index.py

# 3. Ask a question
python rag/query.py "How many co-op work terms do I need to complete?"
```

Retrieval works with no setup.
Generation needs an LLM key: copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY=...`.
Without a key, `query.py` still prints the retrieved chunks so you can inspect retrieval quality on its own — that's deliberate, since retrieval and generation are genuinely separate steps worth being able to test independently.

## Live demo

**[ericdashhuang.github.io/waterloo-first-year-rag](https://ericdashhuang.github.io/waterloo-first-year-rag/)**

A static, browser-only version of this lives on GitHub Pages: see `docs/`.
It computes embeddings client-side (via [transformers.js](https://huggingface.co/docs/transformers.js)), so there's no backend server and no shared API key that a stranger could drain.
Retrieval works immediately with no setup.
To also get a generated written answer (not just the retrieved chunks), you paste your own Anthropic or OpenAI API key into the page — it's used directly from your browser to call the provider and is never sent anywhere else or stored, unless you explicitly opt in to remembering it in that browser's local storage.

## Design decisions worth knowing about

- **Local embedding model, not an API.** Embeddings run through `onnxruntime` on CPU. This keeps the pipeline free to run and testable without any credentials, and it's genuinely fast enough at this corpus size (~400 chunks).
- **Chroma over a hosted vector DB.** It's a single persisted directory, not a service to run or an account to create — the simplest option that's still a real vector database with approximate nearest-neighbor search.
- **Bring-your-own-key for the live demo, not an embedded key.** A public site with an embedded API key is a real cost/abuse risk — anyone who finds it can run up charges. Letting each visitor use their own key removes that risk entirely instead of working around it.
- **Corpus scope.** Limited to official uwaterloo.ca pages under a handful of relevant sections. Reddit and other community sources were deliberately left out for v1, since scraping them reliably needs API credentials this project doesn't depend on.

## Possible next steps

- Add a re-ranking step after retrieval to improve answer quality on ambiguous questions.
- Expand the corpus with more sections (academic advising, clubs, transit).
- Add a small evaluation set of question/answer pairs to measure retrieval accuracy instead of eyeballing it.
