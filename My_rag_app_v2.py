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

if "use_general_knowledge" not in st.session_state:
    st.session_state.use_general_knowledge = False

# SIDEBAR SETTINGS
st.sidebar.header("⚙️ Settings")
st.session_state.use_general_knowledge = st.sidebar.checkbox(
    "🌐 Allow General Knowledge Search",
    value=st.session_state.use_general_knowledge,
    help="If no data found in database, search LLM's general knowledge"
)

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
query = st.chat_input("Ask about IPL Cricket...")

# ASK AI
if query:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    try:
        docs = st.session_state.vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        has_local_data = len(context.strip()) > 0

        # Build prompt based on whether local data exists
        if has_local_data:
            prompt = f"""
You are an expert IPL cricket analyst and commentator.

Use the IPL data context below to answer the question.

Provide:
- Detailed information based on the data
- Statistics and facts
- Player insights if relevant
- Match information

IPL Data Context:
{context}

User Question:
{query}

Answer based on the IPL data:
"""
        else:
            # No local data found
            if st.session_state.use_general_knowledge:
                prompt = f"""
You are an expert IPL cricket analyst and commentator.

The user asked: {query}

No specific information was found in the local IPL database.
You can provide answer based on your general knowledge of IPL cricket.

Answer:
"""
            else:
                prompt = f"""
You are an expert IPL cricket analyst and commentator.

User Question:
{query}

The information about this topic is not available in the IPL database.
Please tell the user that this information is not available in the local database.
"""

        # Call Ollama API
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
            
            # Add note if using general knowledge
            if not has_local_data and st.session_state.use_general_knowledge:
                answer = f"ℹ️ **Using General Knowledge** (No local database match found)\n\n{answer}"

    except requests.exceptions.Timeout:
        answer = "❌ **Timeout Error**: Ollama is taking too long to respond.\n\n**Solutions:**\n1. Make sure Ollama is running: `ollama serve`\n2. Try asking a simpler question\n3. Increase timeout in settings (currently 180 seconds)\n\nCheck that you have Ollama installed and the 'llama3:8b' model downloaded."
    except requests.exceptions.ConnectionError:
        answer = "❌ **Connection Error**: Cannot connect to Ollama.\n\n**Solutions:**\n1. Ensure Ollama is running: `ollama serve` in terminal\n2. Check Ollama is on http://localhost:11434\n3. Verify Ollama is installed on your system"
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    # Add assistant message to history and display
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)