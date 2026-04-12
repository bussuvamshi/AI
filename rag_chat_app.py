import streamlit as st
import requests
import tempfile

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- UI ----------------
st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("RAG application")

# ---------------- CACHE ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def create_vectorstore(chunks):
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# ---------------- SESSION ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(documents)

            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore

            st.success("✅ PDF processed successfully!")

        except Exception as e:
            st.error(f"❌ PDF Processing Error: {str(e)}")


# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------- USER INPUT ----------------
query = st.chat_input("Ask something about your PDF...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        try:
            # Retrieve context
            docs = st.session_state.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Prompt
            prompt = f"""
You are an AI assistant. Answer ONLY from the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

            # ---------------- OLLAMA API ----------------
            url = "http://localhost:11434/api/chat"

            payload = {
                "model": "llama3:8b",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }

            headers = {"Content-Type": "application/json"}

            with st.spinner("Thinking..."):
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=60
                )

                # 🔴 Handle bad responses
                if response.status_code != 200:
                    raise Exception(response.text)

                result = response.json()

                # ✅ Safe parsing
                answer = result.get("message", {}).get("content", "No response from model")

        except Exception as e:
            answer = f"❌ Error: {str(e)}"

        # Display response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
