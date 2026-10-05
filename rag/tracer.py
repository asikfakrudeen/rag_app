import json
import os
import time
import uuid

TRACE_FILE = "traces.jsonl"

def log_trace(query, retrieved_chunks, answer, trace_id=None, latency_ms=0, model="gemini-2.0-flash"):
    """
    Logs the user query, what chunks were fetched, and the final answer 
    into a local JSONL file with a unique trace ID.
    
    If trace_id is None, generates a new one.
    Returns the trace_id that was logged.
    """
    if not trace_id:
        trace_id = str(uuid.uuid4())
        
    top_chunk_source = retrieved_chunks[0]["source"] if retrieved_chunks else "none"
    top_chunk_page = retrieved_chunks[0]["page"] if retrieved_chunks else 0

    trace_record = {
        "trace_id": trace_id,
        "timestamp": time.time(),
        "query": query,
        "answer": answer,
        "latency_ms": round(latency_ms, 2),
        "chunk_count": len(retrieved_chunks),
        "top_chunk_source": top_chunk_source,
        "top_chunk_page": top_chunk_page,
        "fetched_chunks": retrieved_chunks,
        "model": model
    }
    
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record) + "\n")
        
    return trace_id

def search_traces_by_keyword(keyword: str):
    """
    Scans traces.jsonl and returns a list of traces where the keyword
    appears in either the query or the answer. (Case-insensitive)
    """
    if not os.path.exists(TRACE_FILE):
        return []
        
    results = []
    kw = keyword.lower()
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                q = record.get("query", "").lower()
                a = record.get("answer", "").lower()
                if kw in q or kw in a:
                    results.append(record)
            except json.JSONDecodeError:
                pass
    return results

def get_trace_by_id(trace_id: str):
    """
    Scans traces.jsonl and returns the first trace matching trace_id,
    or None if not found.
    """
    if not os.path.exists(TRACE_FILE):
        return None
        
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("trace_id") == trace_id:
                    return record
            except json.JSONDecodeError:
                pass
    return None
