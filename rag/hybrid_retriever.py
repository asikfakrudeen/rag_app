from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from rag.embeddings import create_embedding
import numpy as np

# Load local cross-encoder model for reranking
# We wrap it in a lazy initialization so we don't load it immediately during import if not needed
_reranker = None

def get_reranker():
    """
    Lazily loads the BAAI/bge-reranker-base cross-encoder model.
    This prevents the model from blocking the app startup until it's actually needed.
    
    Example:
        Input: get_reranker()
        Output: <sentence_transformers.cross_encoder.CrossEncoder.CrossEncoder object>
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker

def init_bm25(documents):
    """
    Initializes a BM25 Okapi search index using a list of text documents.
    
    Example:
        Input: init_bm25(["This is the first legal clause", "Here is another contract clause"])
        Output: <rank_bm25.BM25Okapi object>
    """
    # tokenization for BM25: simple lowercase splitting is usually fine for a baseline
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def hybrid_retrieve(collection, bm25_index, all_ids, all_documents, all_metadatas, query, top_k=5, use_hybrid=True, use_rerank=True):
    """
    Retrieves the best matching text chunks for a query using traditional vector search,
    keyword search (BM25), and optional Cross-Encoder reranking.
    
    Example:
        Input: 
            query = "What is the termination notice?"
            top_k = 2
            use_hybrid = True
            hybrid_retrieve(collection, bm25_index, all_ids, all_documents, all_metadatas, query, top_k=2, ...)
            
        Output:
            {
                "ids": [["doc_1_5", "doc_2_1"]],
                "distances": [[0.8243, 0.4532]],
                "documents": [["Termination requires 30 days...", "Mutual termination clause..."]],
                "metadatas": [[{"page": 1, "source": "contract.pdf"}, {"page": 3, "source": "contract.pdf"}]]
            }
    """
    # 1. Semantic (Vector) Search
    query_embedding = create_embedding(query)
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2, # Fetch more context for fusion/reranking
        include=["documents", "metadatas", "distances"]
    )
    
    semantic_ids = semantic_results["ids"][0]
    semantic_docs = semantic_results["documents"][0]
    semantic_metas = semantic_results["metadatas"][0]
    
    if not use_hybrid:
        if not use_rerank:
            return {
                "ids": [semantic_ids[:top_k]],
                "distances": [semantic_results["distances"][0][:top_k]],
                "documents": [semantic_docs[:top_k]],
                "metadatas": [semantic_metas[:top_k]]
            }
        else:
            combined_ids = semantic_ids
            combined_docs = semantic_docs
            combined_metas = semantic_metas
    else:
        # 2. Keyword (BM25) Search
        tokenized_query = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        
        # Get top n indices
        top_n = top_k * 2
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]
        
        bm25_ids = [all_ids[i] for i in top_indices]
        bm25_docs = [all_documents[i] for i in top_indices]
        bm25_metas = [all_metadatas[i] for i in top_indices]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (k + rank)
        k_rf = 60
        fused_scores = {}
        
        # Add semantic scores (ranks are 1-indexed)
        for rank, doc_id in enumerate(semantic_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k_rf + rank + 1))
            
        # Add BM25 scores
        for rank, doc_id in enumerate(bm25_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k_rf + rank + 1))
            
        # Sort by fused score
        sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract combined docs
        combined_ids = []
        combined_docs = []
        combined_metas = []
        
        for doc_id, _ in sorted_fused[:top_k * 2]:
            combined_ids.append(doc_id)
            # Find the actual document text and meta from our lists
            if doc_id in semantic_ids:
                idx = semantic_ids.index(doc_id)
                combined_docs.append(semantic_docs[idx])
                combined_metas.append(semantic_metas[idx])
            else:
                idx = bm25_ids.index(doc_id)
                combined_docs.append(bm25_docs[idx])
                combined_metas.append(bm25_metas[idx])

    # 4. Reranking using Cross-Encoder
    if use_rerank:
        reranker = get_reranker()
        pairs = [[query, doc] for doc in combined_docs]
        rerank_scores = reranker.predict(pairs)
        
        # Sort docs by rerank score
        reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
        
        final_ids = [combined_ids[i] for i in reranked_indices]
        final_docs = [combined_docs[i] for i in reranked_indices]
        final_metas = [combined_metas[i] for i in reranked_indices]
        final_scores = [float(rerank_scores[i]) for i in reranked_indices]
    else:
        final_ids = combined_ids[:top_k]
        final_docs = combined_docs[:top_k]
        final_metas = combined_metas[:top_k]
        final_scores = [0.0] * top_k # Dummy distance if no rerank
        
    return {
        "ids": [final_ids],
        "distances": [final_scores],
        "documents": [final_docs],
        "metadatas": [final_metas]
    }
