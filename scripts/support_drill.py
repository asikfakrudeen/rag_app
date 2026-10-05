"""
Week 11 — Support Drill

Scenario (Track F): A customer complained yesterday that the LLM cited the
wrong termination clause (said 60 days instead of 30). Naturally, nobody
wrote down a trace ID. We just have a vague complaint.

This script:
1. Plants the "bad" trace into traces.jsonl to simulate yesterday's log.
2. Uses the new keyword search feature to find it.
3. Prints the Trace ID and the full context of what the AI saw.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from rag.tracer import log_trace, search_traces_by_keyword

# 1. Plant the bad trace
def plant_bad_trace():
    print("[1] Planting the 'bad' trace from yesterday...")
    
    bad_query = "What is the termination notice period?"
    bad_answer = "According to Section 9.4, the termination notice period is 60 days."
    bad_chunks = [
        {"source": "old_contract_v1.pdf", "page": 4, "distance": 0.2, "text": "Section 9.4: Termination requires 60 days notice."}
    ]
    
    # We call our updated log_trace which returns the ID
    trace_id = log_trace(bad_query, bad_chunks, bad_answer)
    time.sleep(1) # just to ensure timestamp separation
    print(f"    -> Planted trace_id: {trace_id}\n")
    return trace_id

# 2. Support Drill: Find it without knowing the ID
def run_drill():
    print("[2] Initiating Support Drill: Finding the 60 days mistake...")
    
    # We remember the user asked about 'termination' and the bad answer had '60 days'
    results = search_traces_by_keyword("60 days")
    
    if not results:
        print("    -> FAIL: Could not find any trace mentioning '60 days'")
        return
        
    print(f"    -> SUCCESS: Found {len(results)} suspicious trace(s).")
    
    # 3. Analyze the trace
    for i, trace in enumerate(results, 1):
        print(f"\n--- Trace {i} ---")
        print(f"Trace ID: {trace.get('trace_id')}")
        print(f"Timestamp: {trace.get('timestamp')}")
        print(f"Query: {trace.get('query')}")
        print(f"Answer: {trace.get('answer')}")
        
        chunks = trace.get("fetched_chunks", [])
        if chunks:
            top_chunk = chunks[0]
            print(f"\nROOT CAUSE DIAGNOSIS:")
            print(f"Why did the LLM say that? Because it retrieved:")
            print(f"Source: {top_chunk.get('source')} (Page {top_chunk.get('page')})")
            print(f"Text: '{top_chunk.get('text')}'")
        print("-----------------")
        
    print("\n[3] Action Plan:")
    print("Now that we found the trace, we can see the vector DB retrieved 'old_contract_v1.pdf'.")
    print("To stop this from happening again, we will create a regression test that asks")
    print("this exact question and asserts '60 days' is NOT in the answer.")

if __name__ == "__main__":
    plant_bad_trace()
    run_drill()
