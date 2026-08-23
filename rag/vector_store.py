import chromadb
from rag.embeddings import create_embedding

# Initialize ChromaDB client pointing to a local directory so data is saved persistently
chroma_client = chromadb.PersistentClient(path="./data/chroma")


def build_index(chunks, collection_name):
    """
    Build (or rebuild) a ChromaDB collection from the given chunks
    by calculating their embeddings and storing everything in the database.
    
    Example:
        Input: 
            chunks = [{"id": "doc_1_0", "text": "hello", "page": 1, "source": "doc.pdf"}]
            build_index(chunks, "legal_contracts")
            
        Output: <chromadb.api.models.Collection object>
    """

    # Fetch the collection if it exists, or create a brand new one
    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    # Fetch existing data from the collection
    existing = collection.get()

    # Clear out old existing data so we don't accidentally duplicate
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Loop through each chunk sent by the chunker
    for chunk in chunks:

        # Convert the chunk text into a mathematical vector
        embedding = create_embedding(chunk["text"])

        # Insert everything into the ChromaDB collection
        collection.add(
            ids=[chunk["id"]],               # Using the unique ID we generated in chunker.py
            embeddings=[embedding],          # The math
            documents=[chunk["text"]],       # The raw English text
            metadatas=[{                     # Associated metadata (to cite sources later)
                "source": chunk["source"],
                "page": chunk["page"]
            }]
        )

    return collection


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

