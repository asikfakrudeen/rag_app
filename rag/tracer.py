import json
import os
import time

TRACE_FILE = "traces.jsonl"

def log_trace(query, retrieved_chunks, answer):
    """
    Logs the user query, what chunks were fetched, and the final answer 
    into a local JSONL file for downstream error analysis.
    
    Example:
        Input: 
            query = "What is the penalty?"
            retrieved_chunks = [
                {"source": "contract.pdf", "page": 5, "text": "Penalty is $500", "distance": 0.432}
            ]
            answer = "The penalty is $500."
            
            log_trace(query, retrieved_chunks, answer)
            
        Output: 
            None (It writes a new JSON record line to 'traces.jsonl')
    """
    trace_record = {
        "timestamp": time.time(),
        "query": query,
        "fetched_chunks": retrieved_chunks,
        "answer": answer
    }
    
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record) + "\n")
