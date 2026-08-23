import os
import streamlit as st

from rag.pdf_loader import load_pdf
from rag.chunker import create_chunks
from rag.vector_store import build_index, retrieve, get_collection
from rag.generator import generate_answer



st.set_page_config(
    page_title="Legal Contract RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Contract RAG")

st.caption(
    "Ask questions about your amendment documents"
)



with st.sidebar:

    st.header("Document Settings")

    uploaded_files = st.file_uploader(
        "Upload amendment PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("Chunking")

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=1000,
        value=500,
        step=100
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=100,
        step=50
    )

    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=10,
        value=5
    )

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

        st.session_state.collection_name = "legal_contracts"
        st.session_state.indexed = True
        st.session_state.chunk_count = len(chunks)

        st.success(f"Indexed {len(chunks)} chunks.")


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

        results = retrieve(collection, question, top_k)

        answer = generate_answer(question, results)

        st.subheader("Answer")
        st.write(answer)

        st.divider()
        st.subheader("📚 Retrieved Evidence")

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances)
        ):
            with st.expander(
                f"Chunk {i + 1} — "
                f"{metadata['source']} — "
                f"Page {metadata['page']}"
            ):
                st.write(document)
                st.caption(f"Vector distance: {distance:.4f}")
