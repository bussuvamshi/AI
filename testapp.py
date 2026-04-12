import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- UI ----------------
st.set_page_config(page_title="PDF RAG with Llama3", layout="wide")
st.title("📄 Chat with your PDF (Local Llama3)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# ---------------- Load & Process PDF ----------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings (LOCAL - free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("✅ PDF processed successfully!")

    # ---------------- Chat ----------------
    query = st.text_input("Ask something about the PDF")

    if query:
        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt for Llama3
        prompt = f"""
You are an AI assistant. Answer based only on the context below.

Context:
{context}

Question:
{query}

Answer clearly:
"""

        # Call Ollama
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }

        with st.spinner("Thinking..."):
            response = requests.post(url, json=payload)
            result = response.json()

            st.subheader("📌 Answer")
            st.write(result["response"])