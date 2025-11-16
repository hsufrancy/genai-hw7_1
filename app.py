import streamlit as st
from rag_utils import build_index, retrieve, generate_answer

DATA_FOLDER = "data"

st.set_page_config(page_title="HW7 Vector RAG Demo", layout="wide")
st.title("Homework 7: RAG Demo for Data Science Questions")
st.markdown(
    """
This app does the following:

1. Loads documents from the `data/` folder (`pdf`, `txt`, `md`, `docx`).
2. Builds a simple vector-based RAG index using embeddings.
3. Lets you ask data science questions and see both the model's answer and the retrieved chunks.
"""
)

# --- Build index (once, stored in session_state) ---
if "index" not in st.session_state:
    with st.spinner("Building vector index from data/ ..."):
        docs = build_index(DATA_FOLDER)
        st.session_state.index = docs
        st.session_state.num_docs = len(docs)

st.success(f"Index ready! Total chunks: {st.session_state.num_docs}")

# --- QA interface ---
query = st.text_input("Enter your data science question here:", placeholder="E.g., What is Vector RAG?")
top_k = st.slider("Number of contexts to retrieve:", min_value=1, max_value=10, value=5)

if st.button("Ask") and query:
    docs = st.session_state.index
    with st.spinner("Retrieving relevant contexts & generating answers..."):
        hits = retrieve(docs, query, k=top_k)
        answer = generate_answer(query, hits)
    
    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Retrieved Contexts:")
    for i, h in enumerate(hits, start=1):
        with st.expander(f"Context {i} (Score: {h['score']:.4f}) from {h['source']}#{h['chunk_id']}"):
            st.write(h["text"])