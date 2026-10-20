# Known Weak Spots — Legal Contract Review Assistant

> This document is required by the capstone rubric. A demo where nothing ever fails
> hides the real engineering. These are the honest failure modes we found, and how they are mitigated.

---

## 1. API Rate Limits During Indexing

| | |
|---|---|
| **What happens** | Embedding large PDFs quickly exhausts Google's free-tier quota, causing a `RESOURCE_EXHAUSTED` (429) error that crashes the indexing process mid-way. |
| **Reproduction** | Upload a PDF > 50 pages and click "Build Index" without the rate-limit guard. |
| **Mitigation** | `rag/embeddings.py` enforces `time.sleep(1.0)` between every batch call, keeping throughput below the API limit. Chunk size defaults to 1 500 characters to reduce total chunk count. |
| **Remaining risk** | Simultaneous users sharing one API key will still hit the limit. Requires a paid key or a queue mechanism for production. |

---

## 2. Termination Clause Hallucination (Regression — Week 11)

| | |
|---|---|
| **What happens** | When the retrieved context contained a 60-day notice period but the ground-truth clause stated 30 days, the LLM confidently returned the wrong number without any caveat. |
| **How it was found** | Week 11 Support Drill — user reported "it gave a wrong answer". We searched `traces.jsonl` by keyword and isolated the exact trace. |
| **Mitigation** | `scripts/test_termination_regression.py` is now a permanent regression test. The retriever also returns source + page citations so the user can verify. |
| **Root cause** | The wrong chunk was ranked first because "60 days" appeared verbatim in an unrelated amendment; BM25 over-weighted the keyword match. |

---

## 3. Multi-Agent Token Cost Multiplication

| | |
|---|---|
| **What happens** | The orchestrator sends the full retrieved context to **three** specialist agents (ClauseRetriever, RiskAnalyst, SummaryDrafter) sequentially, multiplying token spend by up to 3×. |
| **Reproduction** | Run `scripts/race_multi_vs_single.py` and compare token columns in the output. |
| **Mitigation** | The race script exposes the per-query cost in USD. A routing guard (not yet implemented) could skip specialists for simple factual questions. |
| **When to prefer single agent** | For simple, single-hop factual questions the ReAct single agent is faster and cheaper. Multi-agent is worth the cost only for complex reasoning tasks. |

---

## 4. Empty Index — Unhelpful Error Path

| | |
|---|---|
| **What happens** | If a user asks a question before uploading and indexing any PDFs, the API returns HTTP 400 `"Please build the index first."` — this is technically correct but gives no guidance on what to do. |
| **Mitigation** | The Streamlit UI already shows a warning and disables the Ask button. The React frontend shows a toast notification. |
| **Remaining risk** | Direct API consumers (e.g. `curl /ask`) get a bare 400. A richer error body with a `"hint"` field would improve developer experience. |

---

## 5. No Authentication on API Endpoints

| | |
|---|---|
| **What happens** | `api.py` has CORS set to `allow_origins=["*"]` and no API key check. Any process that can reach port 8000 can call `/build-index` or `/ask` without credentials. |
| **Mitigation** | Acceptable for a local demo/capstone. For production, add an `Authorization` header check (the MCP server already demonstrates this pattern via `MCP_AUTH_TOKEN`). |

---

## 6. Semantic Similarity Misses Exact Clause References

| | |
|---|---|
| **What happens** | Queries like *"What does Section 4(b) say?"* can fail because the vector embedder encodes *meaning*, not section numbers. If the section-number token is diluted in a long chunk, the vector retriever may return an unrelated chunk. |
| **Mitigation** | BM25 hybrid search (Week 6–7) compensates by exact-token matching. Set `use_hybrid=True` (default) to activate it. |
| **Remaining risk** | Very short queries that are purely numeric/structural (e.g. `"Exhibit A"`) may still underperform. |

---

*Last updated: Week 12 Capstone · Track F*
