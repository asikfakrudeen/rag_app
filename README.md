# Legal Contract Review Assistant ⚖️

A production-quality AI assistant for querying legal contracts, built over 12 weeks as Track F of the AI Engineering programme.

Upload a contract PDF, ask plain-English questions, and get cited, grounded answers backed by a hybrid retrieval pipeline and a multi-agent reasoning squad.

---

## What This App Does

- **Retrieval-Augmented Generation (RAG):** Answers questions about contract clauses with exact source citations (document name + page number)
- **Hybrid Search:** Combines dense vector similarity with BM25 keyword search and cross-encoder reranking for high-accuracy retrieval
- **Multi-Agent Reasoning:** An orchestrator delegates to specialist agents (Clause Retriever → Risk Analyst → Summary Drafter) for complex questions
- **Full Observability:** Every query is logged with a unique trace ID for post-incident analysis and cost tracking

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| Node.js | 18 or higher (React frontend only) |
| Google Gemini API key | Free tier works for demos |

---

## Quick Start (3 Steps)

### Step 1 — Install Python dependencies

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
# or, without a venv:
pip install -r requirements.txt
```

### Step 2 — Set your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="AIzaSy...your_actual_key_here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### Step 3 — Launch the app

```powershell
python main.py
```

That's it. Both services start together:

| Service | URL |
|---------|-----|
| Streamlit UI (upload + query) | http://localhost:8501 |
| FastAPI backend (REST API) | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

Press `Ctrl+C` to stop both services.

---

## How to Use

### Build the Index (First Time)

1. Open `http://localhost:8501`
2. In the sidebar, click **Browse files** and upload one or more contract PDFs
3. Click **🔨 Build Index** — wait for the "Indexed N chunks" confirmation
4. The index persists between sessions (stored in ChromaDB)

### Ask a Question

Type a plain-English question in the query box, for example:

> *"What is the termination notice period?"*
> *"Are there any indemnification limits in this agreement?"*
> *"What happens if either party breaches Section 4(b)?"*

Each answer includes:
- The generated answer
- The source chunks that grounded it (document name + page)
- A **Trace ID** for audit and debugging

---

## Run the Evaluation Suite

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

This runs three evaluation stages:
1. Retrieval quality (MRR@5, HitRate@5) across baseline → hybrid → rerank configurations
2. Regression test for the Week 11 termination-clause hallucination
3. Guidance on reading the numbers

For the multi-agent vs single-agent race report:

```powershell
.\.venv\Scripts\python.exe scripts/race_multi_vs_single.py
```

---

## Architecture

```
main.py                   ← Single entry point (starts everything)
│
├── app.py                ← Streamlit UI (upload, query, inspection view)
├── api.py                ← FastAPI backend (/ask, /build-index, /traces)
│
├── rag/
│   ├── document_loader.py   PDF + DOCX ingestion
│   ├── chunker.py           Sliding-window text chunker
│   ├── embeddings.py        Google Gemini text embeddings
│   ├── vector_store.py      ChromaDB persistent vector store
│   ├── hybrid_retriever.py  BM25 + vector + RRF + cross-encoder reranking
│   ├── generator.py         Gemini 2.0 Flash answer generation
│   ├── agent_loop.py        ReAct single-agent loop with MCP tools
│   ├── mcp_host.py          MCP tool host (calls contract_repository server)
│   ├── tracer.py            Request logger → traces.jsonl (with trace IDs)
│   └── multi_agent/
│       ├── orchestrator.py  Manager agent (retrieves context, routes tasks)
│       └── specialists.py   Clause Retriever · Risk Analyst · Summary Drafter
│
├── mcp_server/
│   └── contract_repository.py   MCP tool server (search_clauses, get_chunk)
│
├── scripts/
│   ├── race_multi_vs_single.py  Multi-agent vs single-agent race + report
│   ├── support_drill.py         Find hallucinated answers by keyword in traces
│   ├── test_termination_regression.py  Regression guard for termination clause
│   └── export_traces.py         Export traces.jsonl to CSV for analysis
│
├── evaluate.py           ← End-to-end eval suite (run this for the capstone)
├── KNOWN_WEAK_SPOTS.md   ← Honest failure documentation
└── DEMO_SCRIPT.md        ← Four-part AI Summit demo guide
```

---

## MCP Tool Server (Week 9)

The assistant discovers contract tools via the Model Context Protocol:

```powershell
# Show the MCP handshake
.\.venv\Scripts\python.exe scripts/show_mcp_handshake.py

# Run MCP demo
.\.venv\Scripts\python.exe scripts/demo_mcp.py
```

To expose the MCP server over HTTP for external agents:

```powershell
$env:MCP_AUTH_TOKEN = "a-long-random-token"
python -m mcp_server.contract_repository --http --port 8765
```

---

## Known Weak Spots

This app has real failure modes. They are documented honestly in [`KNOWN_WEAK_SPOTS.md`](./KNOWN_WEAK_SPOTS.md). The key ones:

- **API rate limits** — Large PDFs may hit Google's free-tier quota; mitigated with `time.sleep(1.0)` per embedding batch
- **Hallucination on exact clause numbers** — Found via the Week 11 support drill; regression test now prevents recurrence
- **Multi-agent token cost** — 3× higher than single agent; use the race report to decide which mode to use

---

## Performance Notes

- Chunk size defaults to **1 500 characters** (reduces total chunk count, staying within API limits)
- Rate-limit guard: **1 second delay** between embedding batches
- For large document sets (> 100 pages), a paid Gemini API key is recommended

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google `text-embedding-004` |
| Vector store | ChromaDB (local, persistent) |
| Keyword search | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM |
| Backend API | FastAPI + Uvicorn |
| Streamlit UI | Streamlit |
| React frontend | Vite + React |
| Observability | Custom JSONL tracer (traces.jsonl) |
