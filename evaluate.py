import os
import json
from rag.vector_store import get_collection, get_all_documents
from rag.hybrid_retriever import hybrid_retrieve, init_bm25

# A mocked dataset of queries where generic meaning might fail but exact keywords matter
MOCK_DATA = [
    {
        "query": "What is the penalty for violating ERR-4032?",
        "expected_source": "Vendor_Agreement.pdf"
    },
    {
        "query": "Are there any provisions about mutual termination in section 4(b)?",
        "expected_source": "Master_Service_Agreement.pdf"
    }
]

def mrr_at_k(expected_source, retrieved_metas):
    metas = retrieved_metas[0]
    for i, meta in enumerate(metas):
        if meta.get('source') == expected_source:
            return 1.0 / (i + 1)
    return 0.0

def hit_rate_at_k(expected_source, retrieved_metas):
    metas = retrieved_metas[0]
    for meta in metas:
        if meta.get('source') == expected_source:
            return 1.0
    return 0.0

def run_evaluation(use_hybrid=False, use_rerank=False):
    try:
        collection = get_collection("legal_contracts")
        all_ids, all_docs, all_metas = get_all_documents(collection)
        
        if not all_ids:
            return "No documents indexed. Please upload PDFs and run index build in 'app.py' before running evaluations!"
            
        bm25_index = init_bm25(all_docs)
        
        t_mrr = 0.0
        t_hit = 0.0
        
        for item in MOCK_DATA:
            results = hybrid_retrieve(
                collection=collection,
                bm25_index=bm25_index,
                all_ids=all_ids,
                all_documents=all_docs,
                all_metadatas=all_metas,
                query=item["query"],
                top_k=5,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank
            )
            
            mrr = mrr_at_k(item["expected_source"], results["metadatas"])
            hit = hit_rate_at_k(item["expected_source"], results["metadatas"])
            t_mrr += mrr
            t_hit += hit
            
        avg_mrr = t_mrr / len(MOCK_DATA)
        avg_hit = t_hit / len(MOCK_DATA)
        
        return f"MRR@5: {avg_mrr:.4f} | Hit-Rate@5: {avg_hit:.4f}"
    except Exception as e:
        return f"Error running evaluation: {e}"

if __name__ == "__main__":
    print("--- EVALUATION REPORT ---")
    print("1. Baseline (Semantic Only):")
    print("  ", run_evaluation(use_hybrid=False, use_rerank=False))
    
    print("\n2. Hybrid Search (Semantic + BM25):")
    print("  ", run_evaluation(use_hybrid=True, use_rerank=False))
    
    print("\n3. Hybrid + Reranking (Cross-Encoder):")
    print("  ", run_evaluation(use_hybrid=True, use_rerank=True))
    print("-------------------------\n")
