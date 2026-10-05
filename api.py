import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.document_loader import load_document, SUPPORTED_EXTENSIONS
from rag.chunker import create_chunks
from rag.vector_store import build_index, get_collection, get_all_documents, clear_index
from rag.generator import generate_answer
from rag.tracer import log_trace, get_trace_by_id, search_traces_by_keyword
from rag.hybrid_retriever import hybrid_retrieve, init_bm25
import time

app = FastAPI(title="RAG Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session state equivalent
app_state = {}

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    use_hybrid: bool = True
    use_rerank: bool = True

@app.post("/build-index")
async def api_build_index(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50)
):
    if overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="Overlap must be smaller than chunk size.")
        
    os.makedirs("documents", exist_ok=True)
    all_pages = []
    
    for uploaded_file in files:
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        path = os.path.join("documents", uploaded_file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        pages = load_document(path)
        all_pages.extend(pages)
        
    chunks = create_chunks(all_pages, chunk_size, overlap)
    collection = build_index(chunks, "legal_contracts")
    
    all_ids, all_docs, all_metas = get_all_documents(collection)
    app_state["bm25_index"] = init_bm25(all_docs)
    app_state["all_ids"] = all_ids
    app_state["all_docs"] = all_docs
    app_state["all_metas"] = all_metas
    app_state["indexed"] = True
    
    return {"status": "success", "indexed_chunks": len(chunks)}


@app.delete("/clear-index")
async def api_clear_index():
    """
    Wipes all chunks from the ChromaDB collection and resets the in-memory
    BM25 state. Call this when you want to start fresh with new documents.
    """
    cleared = clear_index("legal_contracts")
    app_state.clear()
    return {"status": "cleared" if cleared else "already_empty"}


@app.post("/ask")
async def api_ask(req: AskRequest):
    start_time = time.time()
    if not app_state.get("indexed"):
        raise HTTPException(status_code=400, detail="Please build the index first.")
        
    collection = get_collection("legal_contracts")
    
    results = hybrid_retrieve(
        collection=collection,
        bm25_index=app_state["bm25_index"],
        all_ids=app_state["all_ids"],
        all_documents=app_state["all_docs"],
        all_metadatas=app_state["all_metas"],
        query=req.question,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,
        use_rerank=req.use_rerank
    )
    
    answer = generate_answer(req.question, results)
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    fetched_chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        fetched_chunks.append({
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 0),
            "distance": float(dist),
            "text": doc
        })
        
    latency_ms = (time.time() - start_time) * 1000
    trace_id = log_trace(req.question, fetched_chunks, answer, latency_ms=latency_ms)
    
    return {
        "answer": answer,
        "evidence": fetched_chunks,
        "trace_id": trace_id
    }

@app.get("/traces/{trace_id}")
async def api_get_trace(trace_id: str):
    trace = get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace

@app.get("/traces")
async def api_search_traces(q: str):
    traces = search_traces_by_keyword(q)
    return {"results": traces}
