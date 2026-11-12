# Assignment 2: My Digital Clone (Hybrid RAG on Hugging Face Spaces)

> **⚠️ This is an AI clone, not the real person.**

**Live app:** https://huggingface.co/spaces/usman-07/usman-digital-clone

A Gradio chatbot that answers questions as me (Usman Tahir), using only documents I wrote about myself.
Retrieval combines **BM25** and **dense embeddings** (`all-MiniLM-L6-v2`) with **Reciprocal Rank Fusion**,
and Google **Gemini** writes the first-person answer from the retrieved chunks only.

## Repository layout

```
digital-clone-space/      # exactly what runs on the Hugging Face Space
  app.py                  # persona prompt, LLM calls, Gradio UI
  rag.py                  # chunking, BM25, dense retrieval, RRF (+ weighted fusion)
  build_index.py          # precomputes embeddings.npy + chunks.json
  evaluate.py             # hit@3 per mode + report table
  requirements.txt
  README.md               # Space config + the one-page report
  data/                   # knowledge base (9 first-person documents)
  chunks.json, embeddings.npy
deploy.py                 # uploads digital-clone-space/ to the Space; keys go in as Space secrets
```

The full report (knowledge base, 5-question comparison, failure case, improvement) is in
[digital-clone-space/README.md](digital-clone-space/README.md).

## Run locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r digital-clone-space/requirements.txt
cd digital-clone-space
python build_index.py            # only after editing data/
GEMINI_API_KEY=your-key python app.py
```

API keys are never stored in this repo. Locally they come from a `.env` file (git-ignored), and on the
Space they are stored as secrets.
