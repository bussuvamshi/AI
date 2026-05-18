import os
import streamlit as st
import requests
import json
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# CONFIG
PDF_FOLDER = r"C:\Users\bussu\Documents\IPL DB"
DB_FOLDER = "vectorstore"
METADATA_FILE = "db_metadata.json"
OLLAMA_API = "http://localhost:11434/api/chat"
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".html", ".doc"]

# CACHE
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def create_vectorstore(chunks):
    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)

# GET LIST OF SUPPORTED FILES
def get_supported_files(folder_path):
    """Get list of all supported document files in folder"""
    if not os.path.exists(folder_path):
        return []
    
    files = []
    for file in os.listdir(folder_path):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            files.append(file)
    return sorted(files)

# LOAD DOCUMENT BY FILE TYPE
def load_document(file_path):
    """Load document based on file extension"""
    file_lower = file_path.lower()
    documents = []
    
    try:
        if file_lower.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_lower.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        elif file_lower.endswith(('.docx', '.doc')):
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
        elif file_lower.endswith('.html'):
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
    
    return documents

# SIMPLE GREETING DETECTION
# Prevent casual greetings like "Hi" from triggering large RAG context lookups
# and keep the assistant response short and friendly.
def is_simple_greeting(query):
    normalized = query.strip().lower()
    greetings = [
        "hi", "hello", "hey", "hi there", "hello there",
        "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "whats up"
    ]
    if len(normalized) > 40:
        return False
    return any(normalized == greeting or normalized.startswith(greeting + " ") or normalized.endswith(" " + greeting) for greeting in greetings)


def is_affirmative(query):
    normalized = query.strip().lower()
    return normalized in ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "please do", "please"]


def is_negative(query):
    normalized = query.strip().lower()
    return normalized in ["no", "n", "nope", "not now", "don't", "do not", "nah"]

# LOAD METADATA
def load_metadata():
    """Load previously processed files list"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except:
            return {"processed_files": [], "last_updated": None}
    return {"processed_files": [], "last_updated": None}

# SAVE METADATA
def save_metadata(processed_files):
    """Save list of processed files"""
    metadata = {
        "processed_files": processed_files,
        "last_updated": datetime.now().isoformat()
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# GET NEW FILES
def get_new_files(folder_path):
    """Return list of new files not in database"""
    current_files = get_supported_files(folder_path)
    metadata = load_metadata()
    processed_files = metadata.get("processed_files", [])
    
    new_files = [f for f in current_files if f not in processed_files]
    return new_files

# LOAD NEW DOCUMENTS ONLY
def load_new_documents(folder_path, new_files):
    """Load only new document files"""
    documents = []
    
    for file in new_files:
        full_path = os.path.join(folder_path, file)
        try:
            docs = load_document(full_path)
            documents.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Couldn't load {file}: {str(e)}")
    
    return documents

# LOAD ALL DOCUMENTS FROM SUPPORTED FILES
# This supports .pdf, .txt, .docx, .doc, and .html files in the IPL DB folder.
def load_all_documents(folder_path):
    documents = []
    for file in get_supported_files(folder_path):
        full_path = os.path.join(folder_path, file)
        try:
            docs = load_document(full_path)
            documents.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Couldn't load {file}: {str(e)}")
    if not documents:
        raise FileNotFoundError(f"No supported documents found in '{folder_path}'")
    return documents

# LOAD ALL PDF FILES
def load_all_pdfs(folder_path):

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found")

    documents = []
    pdf_count = 0

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_count += 1
            full_path = os.path.join(folder_path, file)
            try:
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.warning(f"⚠️ Couldn't load {file}: {str(e)}")

    if pdf_count == 0:
        raise FileNotFoundError(f"No PDF files found in '{folder_path}'")

    return documents

# CREATE VECTOR DATABASE
def create_vector_db():
    docs = load_all_documents(PDF_FOLDER)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    vectorstore.save_local(DB_FOLDER)
    
    # Save metadata
    current_files = get_supported_files(PDF_FOLDER)
    save_metadata(current_files)

    return vectorstore

# UPDATE VECTOR DATABASE WITH NEW FILES
def update_vector_db(vectorstore):
    """Add new documents to existing vector database"""
    new_files = get_new_files(PDF_FOLDER)
    
    if not new_files:
        return vectorstore, 0
    
    # Load only new files
    new_docs = load_new_documents(PDF_FOLDER, new_files)
    
    if not new_docs:
        return vectorstore, 0
    
    # Split new documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_new_docs = text_splitter.split_documents(new_docs)
    
    # Add to existing vectorstore
    embeddings = load_embeddings()
    vectorstore.add_documents(split_new_docs)
    vectorstore.save_local(DB_FOLDER)
    
    # Update metadata
    all_files = get_supported_files(PDF_FOLDER)
    save_metadata(all_files)
    
    return vectorstore, len(new_files)

# LOAD EXISTING VECTOR DB
def load_vector_db():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        DB_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# STREAMLIT UI
st.set_page_config(page_title="IPL Cricket RAG", layout="wide")
st.title("IPL Cricket AI Assistant")
st.write("Ask me anything about IPL cricket...")

# SESSION STATE
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

# LOAD VECTOR DB & AUTO UPDATE
try:
    if not os.path.exists(DB_FOLDER):
        with st.spinner("Reading documents and creating vector DB..."):
            vectorstore = create_vector_db()
            st.session_state.vectorstore = vectorstore
            st.session_state.db_initialized = True
            st.success("✅ Vector database created!")
    else:
        if st.session_state.vectorstore is None:
            vectorstore = load_vector_db()
            st.session_state.vectorstore = vectorstore
            st.session_state.db_initialized = True
            
            # CHECK FOR NEW FILES & AUTO UPDATE
            new_files = get_new_files(PDF_FOLDER)
            if len(new_files) > 0:
                with st.spinner(f"Auto-updating database with {len(new_files)} new file(s)..."):
                    vectorstore, updated_count = update_vector_db(vectorstore)
                    st.session_state.vectorstore = vectorstore
                    if updated_count > 0:
                        st.success(f"✅ Auto-updated! Added {updated_count} new file(s)")
except Exception as e:
    st.error(f"❌ Error loading database: {str(e)}")
    st.stop()

# SHOW DATABASE STATUS
supported_files = get_supported_files(PDF_FOLDER)
metadata = load_metadata()
processed = metadata.get("processed_files", [])

# DISPLAY CHAT HISTORY
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# USER INPUT
query = st.chat_input("Ask about IPL Cricket...", key="query_input")

# Auto-focus the prompt input on initial load and after rerender.
st.markdown(
    """
    <script>
    const focusPrompt = () => {
      const selector = 'input[placeholder="Ask about IPL Cricket..."], textarea[placeholder="Ask about IPL Cricket..."]';
      const el = document.querySelector(selector);
      if (el) {
        el.focus();
        el.scrollIntoView({behavior: 'smooth', block: 'center'});
        return true;
      }
      return false;
    };

    const intervalId = window.setInterval(() => {
      if (focusPrompt()) {
        window.clearInterval(intervalId);
      }
    }, 200);

    window.addEventListener('load', () => {
      focusPrompt();
    });

    document.addEventListener('readystatechange', () => {
      if (document.readyState === 'complete') {
        focusPrompt();
      }
    });

    window.setTimeout(() => {
      focusPrompt();
      window.clearInterval(intervalId);
    }, 3000);
    </script>
    """,
    unsafe_allow_html=True,
)

# ASK AI
if query:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    try:
        if is_simple_greeting(query):
            answer = "Hi! 👋 I’m your IPL Cricket assistant. Ask me anything about IPL teams, players, matches, or stats."

        elif st.session_state.get("await_global_permission", False):
            if is_affirmative(query):
                prompt = f"""
You are an expert IPL cricket analyst and commentator.
Answer the user's question using your global knowledge only.
Do not use, reference, or transfer any content from the internal IPL database.
Answer the question clearly and concisely.

User Question:
{st.session_state.get('pending_query', query)}
"""
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
                        OLLAMA_API,
                        json=payload,
                        headers=headers,
                        timeout=180
                    )

                    if response.status_code != 200:
                        raise Exception(f"Ollama API error: {response.text}")

                    result = response.json()
                    answer = result.get("message", {}).get("content", "No response from model")

                st.session_state.await_global_permission = False
                st.session_state.pending_query = None

            elif is_negative(query):
                answer = "Okay, I will not use global sources. Please ask another IPL question or a different query."
                st.session_state.await_global_permission = False
                st.session_state.pending_query = None
            else:
                answer = "Please answer 'yes' or 'no': can I check the global source for your question?"

        else:
            docs = st.session_state.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            has_local_data = len(context.strip()) > 0

            if has_local_data:
                prompt = f"""
You are an expert IPL cricket analyst and commentator.
Use only the IPL data context below to answer the question.
Do not invent extra details or use unrelated data.
If the context is not enough to answer the question, respond exactly: NO_MATCH
Answer clearly and concisely.

IPL Data Context:
{context}

User Question:
{query}
"""
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
                        OLLAMA_API,
                        json=payload,
                        headers=headers,
                        timeout=180
                    )

                    if response.status_code != 200:
                        raise Exception(f"Ollama API error: {response.text}")

                    result = response.json()
                    answer = result.get("message", {}).get("content", "No response from model")

                if answer is None:
                    answer = "No response from model"
                cleaned = answer.strip().upper()
                if cleaned == "NO_MATCH" or cleaned.startswith("NO_MATCH"):
                    answer = (
                        "Sorry, I don't have the proper information for your question in the internal database. "
                        "Can I try to answer it from the global source? Please reply 'yes' or 'no'."
                    )
                    st.session_state.await_global_permission = True
                    st.session_state.pending_query = query

            else:
                answer = (
                    "Sorry, I don't have the proper information for your question in the internal database. "
                    "Can I try to answer it from the global source? Please reply 'yes' or 'no'."
                )
                st.session_state.await_global_permission = True
                st.session_state.pending_query = query

    except requests.exceptions.Timeout:
        answer = "❌ **Timeout Error**: Ollama is taking too long to respond.\n\n**Solutions:**\n1. Make sure Ollama is running: `ollama serve`\n2. Try asking a simpler question\n3. Increase timeout in settings (currently 180 seconds)\n\nCheck that you have Ollama installed and the 'llama3:8b' model downloaded."
    except requests.exceptions.ConnectionError:
        answer = "❌ **Connection Error**: Cannot connect to Ollama.\n\n**Solutions:**\n1. Ensure Ollama is running: `ollama serve` in terminal\n2. Check Ollama is on http://localhost:11434\n3. Verify Ollama is installed on your system"
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    # Add assistant message to history and display
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)