# IPL Cricket RAG Application

A Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and Ollama for querying IPL cricket data and documents.

## 📁 Project Structure

```
AI/
├── My_rag_app_v2.py          # Main IPL Cricket RAG app (latest)
├── requirements.txt           # Python dependencies
├── Rag_appv1/                # First RAG version
│   ├── rag_chat_app.py       # PDF upload-based RAG chat
│   ├── app.py                # OpenAI integration example
│   └── requirements.txt
├── Rag_appv2/                # Second RAG version
│   ├── My_rag_app_v2.py      # Auto-updating RAG app
│   ├── db_metadata.json      # Database metadata (auto-generated)
│   ├── vectorstore/          # FAISS vector database (auto-generated)
│   └── requirements.txt
└── env/                       # Python virtual environment
```

## 🚀 Features

### My_rag_app_v2.py (Latest & Recommended)
- 📚 **Multi-format support**: PDF, TXT, DOCX, HTML files
- 🔄 **Auto-update**: Automatically detects and indexes new documents
- 💬 **Conversational Chat**: ChatGPT-like interface with full conversation history
- 🌐 **General Knowledge Fallback**: Uses LLM knowledge when local DB has no match
- 📊 **Vector Database**: FAISS for fast semantic search
- 🤖 **Ollama Integration**: Uses local Llama3 model for responses
- ⚡ **Efficient Caching**: Streamlit caching for faster performance

### Rag_appv1/rag_chat_app.py
- 📤 **File Upload**: Upload PDFs directly in the UI
- 💾 **Session Management**: Keeps track of chat history per session
- 🔍 **Similarity Search**: Find relevant document chunks

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Install Llama3 model: `ollama pull llama3:8b`

3. **Data Folder**
   - Create or point to: `C:\Users\bussu\Documents\IPL DB`
   - Add your PDF, TXT, DOCX, or HTML files

## 🔧 Installation

### Step 1: Clone/Copy the Project
```bash
cd C:\Users\bussu\MyPracticalsVScode\AI
```

### Step 2: Create Virtual Environment
```powershell
python -m venv env
.\env\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

## ▶️ Running the Application

### 1. Start Ollama Server (Required!)
Open a new PowerShell terminal and run:
```powershell
ollama serve
```

Keep this terminal open while running the app.

### 2. Run the RAG App
In another terminal:
```powershell
cd C:\Users\bussu\MyPracticalsVScode\AI
.\env\Scripts\activate
streamlit run My_rag_app_v2.py
```

### 3. Access the App
- Open your browser to: `http://localhost:8501`
- Start asking questions about your IPL data!

## 📝 Configuration

### Update Data Folder Path
Edit `My_rag_app_v2.py`:
```python
PDF_FOLDER = r"C:\Your\Path\To\IPL DB"
```

### Adjust LLM Model
Edit the model in the Ollama API call (default: `llama3:8b`):
```python
"model": "llama3:8b"  # or llama2, mistral, etc.
```

### Modify Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Increase for larger chunks
    chunk_overlap=100
)
```

## 🔑 Environment Variables

For Rag_appv1/app.py (OpenAI integration):
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

## 📊 Supported File Formats

- **PDF** - `.pdf` (via PyPDFLoader)
- **Text** - `.txt` (via TextLoader)
- **Word** - `.docx`, `.doc` (via UnstructuredWordDocumentLoader)
- **HTML** - `.html` (via UnstructuredHTMLLoader)

## 💡 How It Works

1. **Document Loading**: Reads all supported file types from the folder
2. **Text Splitting**: Splits documents into smaller chunks (500 tokens)
3. **Embedding**: Converts chunks to embeddings using HuggingFace
4. **Vector Storage**: Stores embeddings in FAISS database
5. **Query Processing**:
   - User asks a question
   - Finds most relevant document chunks
   - Sends context + question to Ollama Llama3
   - Returns answer from LLM

## 🆘 Troubleshooting

### Error: "HTTPConnectionPool(host='localhost', port=11434): Read timed out"
**Solution**: 
- Make sure Ollama is running: `ollama serve`
- Check Ollama is installed: `ollama --version`
- Download model if missing: `ollama pull llama3:8b`

### Error: "No PDF files found"
**Solution**:
- Create folder: `C:\Users\bussu\Documents\IPL DB`
- Add your document files to this folder
- Refresh the Streamlit app

### Error: "Cannot connect to Ollama"
**Solution**:
- Ollama must be running in another terminal with `ollama serve`
- Verify it's listening on `http://localhost:11434`
- Check Ollama installation

### App is slow
**Solution**:
- Reduce `chunk_size` in the code
- Use smaller model: `ollama pull mistral:7b`
- Reduce `k=5` to `k=3` in similarity_search

## 📚 Dependencies

See `requirements.txt` for full list. Key packages:
- **streamlit** - Web UI framework
- **langchain** - LLM orchestration
- **faiss-cpu** - Vector similarity search
- **requests** - HTTP client for Ollama
- **pypdf** - PDF processing
- **langchain-huggingface** - Hugging Face embeddings

## 🔄 Workflow Example

1. Add IPL cricket documents to your folder
2. Run the app
3. App automatically creates vector database
4. Ask: "Who is the highest scorer in IPL?"
5. App searches documents + uses LLM knowledge
6. Returns answer with relevant stats

## 📖 Version Comparison

| Feature | Rag_appv1 | My_rag_app_v2 |
|---------|-----------|---------------|
| Multi-format | ❌ | ✅ |
| Auto-update | ❌ | ✅ |
| Chat history | ✅ | ✅ |
| General knowledge | ❌ | ✅ |
| Web upload | ✅ | ❌ |
| Folder monitoring | ❌ | ✅ |

## 🤝 Contributing

Feel free to modify and extend the application!

## 📄 License

This project is open source and available for personal and educational use.

## 🙋 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify Ollama is running
3. Check your data folder path
4. Review the console logs in Streamlit

---

**Last Updated**: May 15, 2026
**Version**: 2.0
