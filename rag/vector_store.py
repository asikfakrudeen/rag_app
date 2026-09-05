import chromadb
from rag.embeddings import create_embedding

# Initialize ChromaDB client pointing to a local directory so data is saved persistently
chroma_client = chromadb.PersistentClient(path="./data/chroma")


def build_index(chunks, collection_name, append=True):
    """
    Build or update a ChromaDB collection from the given chunks.

    Args:
        append (bool): 
            True  → Keep existing chunks from OTHER documents. Only remove
                    chunks that belong to the same source files being re-indexed
                    (prevents duplicates on re-upload without wiping other docs).
            False → Wipe the entire collection before inserting (full rebuild).
    """

    collection = chroma_client.get_or_create_collection(name=collection_name)

    if append:
        # Find which source files are present in the new chunks
        new_sources = {chunk["source"] for chunk in chunks}

        # Fetch all existing data and remove only chunks from those same sources
        existing = collection.get(include=["metadatas"])
        ids_to_remove = [
            doc_id
            for doc_id, meta in zip(existing["ids"], existing["metadatas"])
            if meta.get("source") in new_sources
        ]
        if ids_to_remove:
            collection.delete(ids=ids_to_remove)
    else:
        # Full wipe — remove every chunk in the collection
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    # Embed and insert new chunks
    for chunk in chunks:
        embedding = create_embedding(chunk["text"])
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "page": chunk["page"]
            }]
        )

    return collection


def clear_index(collection_name):
    """
    Wipes all data from a collection. Used by the /clear-index API endpoint.
    """
    try:
        collection = chroma_client.get_collection(collection_name)
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        return True
    except Exception:
        return False


def retrieve(collection, question, top_k):
    """
    Converts a user's question into an embedding and queries the Chroma database 
    to retrieve the most relevant text chunks.
    
    Example:
        Input: retrieve(collection, "What is the notice period?", top_k=2)
        Output:
            {
                "ids": [["doc.pdf_1_0", "doc.pdf_4_2"]],
                "distances": [[0.123, 0.456]],
                "documents": [["Notice is 30 days...", "Termination requires notice..."]],
                "metadatas": [[{"page": 1, "source": "doc.pdf"}, {"page": 4, "source": "doc.pdf"}]]
            }
    """

    # Calculate the embedding (math representation) for the user's question
    query_embedding = create_embedding(question)

    # Query ChromaDB, asking for 'top_k' nearest neighboring chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results


def get_collection(collection_name):
    """
    Get an existing ChromaDB collection by name.
    
    Example:
        Input: get_collection("legal_contracts")
        Output: <chromadb.api.models.Collection object>
    """
    
    return chroma_client.get_collection(collection_name)

def get_all_documents(collection):
    """
    Fetches all documents and their IDs from a given collection to build local indexes like BM25.
    """
    try:
        existing = collection.get()
        return existing["ids"], existing["documents"], existing["metadatas"]
    except Exception:
        return [], [], []

