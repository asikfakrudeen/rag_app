from rag.vector_store import get_collection
from rag.hybrid_retriever import hybrid_retrieve
from rag.generator import generate_answer
from rag.tracer import log_trace

def run_fixed_workflow(question: str, app_state: dict):
    """
    Executes the deterministic, linear RAG pipeline.
    """
    if not app_state.get("indexed"):
        raise ValueError("Please build the index first.")
        
    collection = get_collection("legal_contracts")
    
    results = hybrid_retrieve(
        collection=collection,
        bm25_index=app_state["bm25_index"],
        all_ids=app_state["all_ids"],
        all_documents=app_state["all_docs"],
        all_metadatas=app_state["all_metas"],
        query=question,
        top_k=5,
        use_hybrid=True,
        use_rerank=True
    )
    
    answer = generate_answer(question, results)
    
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
        
    log_trace(question, fetched_chunks, answer)
    
    return {
        "answer": answer,
        "evidence": fetched_chunks
    }
