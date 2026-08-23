import os
import streamlit as st

from rag.pdf_loader import load_pdf
from rag.chunker import create_chunks
from rag.vector_store import build_index, get_collection, get_all_documents
from rag.generator import generate_answer
from rag.tracer import log_trace
from rag.hybrid_retriever import hybrid_retrieve, init_bm25

st.set_page_config(
    page_title="Legal Contract RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Contract RAG & Inspection")

st.caption("Ask questions about your amendment documents and debug retrieval")

with st.sidebar:
    st.header("Document Settings")
    uploaded_files = st.file_uploader(
        "Upload amendment PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("Chunking")
    chunk_size = st.slider("Chunk size", 100, 1000, 500, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 100, 50)
    top_k = st.slider("Top-K", 1, 10, 5)
    
    st.divider()
    st.header("Retrieval Settings")
    use_hybrid = st.toggle("Enable BM25 Hybrid Search", value=True)
    use_rerank = st.toggle("Enable Cross-Encoder Reranking", value=True)

    build_button = st.button("🔨 Build Index")


if build_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    elif overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
    else:
        all_pages = []
        for uploaded_file in uploaded_files:
            os.makedirs("documents", exist_ok=True)
            path = os.path.join("documents", uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pages = load_pdf(path)
            all_pages.extend(pages)

        chunks = create_chunks(all_pages, chunk_size, overlap)
        collection = build_index(chunks, "legal_contracts")
        
        all_ids, all_docs, all_metas = get_all_documents(collection)
        st.session_state.bm25_index = init_bm25(all_docs)
        st.session_state.all_ids = all_ids
        st.session_state.all_docs = all_docs
        st.session_state.all_metas = all_metas
        
        st.session_state.collection_name = "legal_contracts"
        st.session_state.indexed = True
        st.session_state.chunk_count = len(chunks)

        st.success(f"Indexed {len(chunks)} chunks and built BM25 index.")


st.header("Ask the Contract")

question = st.text_input(
    "Enter your question",
    placeholder="What is the amended termination notice period?"
)

if st.button("🔍 Ask"):
    if not question:
        st.warning("Please enter a question.")
    elif "indexed" not in st.session_state:
        st.warning("Please build the index first.")
    else:
        collection = get_collection("legal_contracts")

        results = hybrid_retrieve(
            collection=collection,
            bm25_index=st.session_state.bm25_index,
            all_ids=st.session_state.all_ids,
            all_documents=st.session_state.all_docs,
            all_metadatas=st.session_state.all_metas,
            query=question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank
        )

        answer = generate_answer(question, results)

        # Log the trace for error analysis
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

        st.subheader("Inspection View")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Answer Output")
            st.info(f"**Query:** {question}")
            st.success(answer)
            
        with col2:
            st.markdown("### Fetched Evidence")
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances)
            ):
                with st.expander(
                    f"Chunk {i + 1} — "
                    f"{metadata['source']} — "
                    f"Page {metadata['page']} (Score: {distance:.4f})"
                ):
                    st.write(document)
