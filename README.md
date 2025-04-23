# 📄 Chat with PDF

**Chat with PDF** is a powerful AI-driven Streamlit application that allows users to upload PDF documents and ask natural language questions about their content. The system leverages LangChain, FAISS, and Google’s Gemini Pro model to return context-aware, accurate answers from your uploaded files.

---

## ✨ Features

- 📄 Upload and process multiple PDFs
- 🔍 Extracts and chunks PDF content using LangChain
- 🧠 Embedding with Google Generative AI
- 🗂️ Similarity search using FAISS
- 💬 Answer questions with Gemini Pro (ChatGoogleGenerativeAI)
- 💾 Save and reload chat sessions
- 📝 Optional feedback collection

---

## 🛠️ Tech Stack

### 🔹 Frontend
- Streamlit

### 🔹 Backend / Processing
- PyPDF2 – for PDF text extraction  
- LangChain – for chunking, embedding, and QA chain  
- FAISS – for vector similarity search  
- Google Generative AI – for embeddings and chat (Gemini Pro)  
- dotenv – to manage environment variables

---

## ✅ Prerequisites

- Python 3.9+
- pip
- Google Generative AI API Key

---

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create a `.env` file in the root directory:**
```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🗂️ Project Structure

```
chat-with-pdf/
├── main.py                 # Main Streamlit application
├── faiss_index/            # Local FAISS vector storage
├── .env                    # Google API Key configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🏗️ Architecture Flow

```text
User Uploads PDFs
        ↓
  Text Extraction (PyPDF2)
        ↓
Chunking (LangChain Splitter)
        ↓
 Embedding (Gemini via LangChain)
        ↓
 Vector Storage (FAISS)
        ↓
  Similarity Search (FAISS)
        ↓
QA Chain using Gemini Pro (LangChain)
        ↓
     Answer Display (Streamlit)
```

---

## 🚀 Running the Application

1. **Launch the app using Streamlit:**
```bash
streamlit run main.py
```

2. Open your browser and go to:
```
http://localhost:8501
```

---

## 🧑‍💻 Usage

1. Upload one or more PDFs using the uploader.
2. Click **“Process PDFs”** to split and store them in a vector store.
3. Ask questions separated by commas.
4. View answers powered by Gemini Pro using document context.
5. Save and load sessions as needed.
6. Submit feedback if desired.

---

## 🧪 Functional Components

### 🔍 Text Processing
- Chunking large documents with overlap
- Extracting plain text from each PDF page

### 🤖 AI Interaction
- Embedding via `GoogleGenerativeAIEmbeddings`
- Response generation via `ChatGoogleGenerativeAI`

### 🧠 Context Search
- FAISS vector DB similarity match
- Top-matching document chunks used for QA

---

## 🔐 Security Considerations

- Supports only safe file types (PDF)
- No cloud upload – documents processed locally
- API keys managed through `.env`
- Uses LangChain's safe deserialization

---

## 🧠 Future Enhancements

- OCR support for scanned PDFs (using Tesseract or similar)
- Export conversation to PDF
- Allow model switching: Gemini, OpenAI, Claude
- UI enhancements for mobile devices
- Semantic search visualization

---

## 🤝 Contributing

1. Fork the repository  
2. Create a new branch: `git checkout -b feature/YourFeature`  
3. Commit your changes: `git commit -m 'Add some feature'`  
4. Push to the branch: `git push origin feature/YourFeature`  
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/)  
- [Google Generative AI](https://ai.google.dev)  
- [Streamlit](https://streamlit.io/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [PyPDF2](https://pypi.org/project/PyPDF2/)

