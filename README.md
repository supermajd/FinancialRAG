# FinancialRAG — Private, local document Q&A

Ask questions about financial documents using a **local LLM**, your data never
leaves your computer. This repo is built with more than a decade of
H&M Group's public filings (annual reports, AGM/EGM documents, governance and
sustainability reports), so you can clone it and ask questions in minutes.

Every answer comes with **citations**: the source document, section, year, and a
similarity score.

**How it works:** PDFs → Markdown (layout-aware, tables preserved) → structure-aware
chunking → hybrid search (FAISS vectors + BM25 keywords) → local LLM answers with
citations.

---

## Requirements

| Tool | Why | Get it |
|---|---|---|
| Python 3.10+ | backend + frontend server | python.org |
| Ollama | runs the local LLM | https://ollama.com |

---

## Quick start (~5 minutes)

The H&M index is included — no data download or indexing needed.

**1. Clone and install:**

```bash
git clone <repo-url>
cd <repo>
python3 -m venv RAG
source RAG/bin/activate        # Windows: RAG\Scripts\activate
pip install -r requirements.txt
```

**2. Get the LLM** (leave Ollama running):

```bash
ollama pull qwen2.5:14b-instruct
```

**3. Start the backend** — terminal 1:

```bash
source RAG/bin/activate
uvicorn app.api:app --port 8000
```

> First start downloads the embedding model (~500 MB, one time). Wait for `Ready`.

**4. Start the frontend** — terminal 2 (from the repo root):

```bash
python3 -m http.server 8080
```

Then open **http://localhost:8080/FinancialRAG.html** in your browser.

**5. Ask questions:**

- *"What was the total H&M Group revenue in 2025?"*
- *"Who was elected to the board at the 2025 AGM?"*
- *"What dividend did the board propose in 2019?"*

Or hit the API directly:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What was the total H&M Group revenue in 2025?"}'
```

> ⚠️ **Known issue (fix in progress):** the model sometimes fails to invoke the
> search tool on the first attempt — if an answer comes back without sources,
> retry the question a couple of times.

---

## Rebuilding the index from scratch (optional)

Only needed if you change the corpus or the chunking.

**1. Download and convert the PDFs:**

```bash
python data/download_data.py
```

Downloads every PDF listed in `data/documents.csv` and converts it to Markdown
with `marker`.

> ⏳ **This is slow by design.** Marker runs ML models on every page to keep
> financial tables intact — that's what makes exact-figure answers possible.
> ~288 documents on CPU can take hours. It's resumable: re-running skips
> anything already converted.

**2. Rebuild the database and indexes:**

```bash
python -m app.setup
```

Chunks every document (tables kept whole), embeds them, and rebuilds the
FAISS + BM25 indexes. Re-run after adding documents.

---

## Using your own documents

Add each document's URL and metadata (category, year, title) as a row in
`data/documents.csv`, then run the two rebuild steps above. Everything runs
locally — no document content is sent anywhere.

---

## Troubleshooting

- **Backend exits with "No FAISS index found"** — the index is missing; run the
  rebuild steps, or re-pull the repo (the prebuilt index ships in `data/index/`).
- **Frontend loads but requests fail** — the backend must run on port **8000**,
  and the frontend must be served from port **8080** (the API's CORS allowlist
  covers localhost:8080, 5173, and 3000).
- **First answer is slow** — the embedding model downloads once on first start;
  local LLM speed depends on your hardware.
