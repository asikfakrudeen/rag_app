import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from rag.vector_store import get_collection, get_all_documents
from rag.hybrid_retriever import init_bm25
from rag.fixed_workflow import run_fixed_workflow
from rag.agent_loop import run_agent_loop

def setup_app_state():
    print("[SETUP] Loading index and preparing App State...")
    collection = get_collection("legal_contracts")
    # Quick check if it's empty
    if collection.count() == 0:
        print("WARNING: Collection is empty. Re-index a document using app.py.")
        return None
        
    all_ids, all_docs, all_metas = get_all_documents(collection)
    bm25_index = init_bm25(all_docs)
    
    app_state = {
        "indexed": True,
        "all_ids": all_ids,
        "all_docs": all_docs,
        "all_metas": all_metas,
        "bm25_index": bm25_index
    }
    return app_state

def main():
    app_state = setup_app_state()
    if not app_state:
        return
        
    test_question = "What happens if there's a breach of contract according to the terms?"
    
    print("\n" + "="*50)
    print(f"QUESTION: {test_question}")
    print("="*50 + "\n")
    
    # 1. FIXED WORKFLOW
    print("[RACING] ➡️ Fixed Workflow Baseline")
    start_t = time.time()
    try:
        fixed_result = run_fixed_workflow(test_question, app_state)
        fixed_time = time.time() - start_t
        print(f"  Time taken: {fixed_time:.2f}s")
        print(f"  Answer: {fixed_result['answer']}\n")
    except Exception as e:
        print(f"  Failed: {e}\n")
        
    # 2. ReAct AGENT
    print("[RACING] ➡️ Pure Python ReAct Agent")
    start_t = time.time()
    try:
        agent_result = run_agent_loop(test_question, app_state, max_iterations=5)
        agent_time = time.time() - start_t
        print(f"  Time taken: {agent_time:.2f}s")
        print(f"  Answer: {agent_result['answer']}")
        print("\n  🧠 Agent Memory Trace:")
        print(agent_result['memory'])
    except Exception as e:
        print(f"  Failed: {e}\n")

    # 3. TEST LONG TERM MEMORY (Agent Only)
    followup_question = "What was the previous question I just asked you? Answer using your long-term memory."
    print("\n" + "="*50)
    print(f"FOLLOW-UP QUESTION: {followup_question}")
    print("="*50 + "\n")
    
    print("[RACING] ➡️ Pure Python ReAct Agent (Follow-up)")
    try:
        agent_result2 = run_agent_loop(followup_question, app_state, max_iterations=5)
        print(f"  Answer: {agent_result2['answer']}")
    except Exception as e:
        print(f"  Failed: {e}\n")

if __name__ == "__main__":
    main()
