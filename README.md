project:
  title: "📄 Chat with PDF"
  description: >
    This project is a PDF-based Conversational AI system built using Streamlit, LangChain, and Google Gemini Pro.
    It allows users to upload PDF documents, ask questions, and receive accurate, context-based answers using advanced language models.

features:
  - Upload and process multiple PDF files
  - Extract and chunk PDF content using LangChain
  - Generate embeddings using Google Generative AI
  - Store vectors in a FAISS vector database
  - Answer user questions with Gemini Pro (ChatGoogleGenerativeAI)
  - Save and load sessions
  - Collect user feedback

tech_stack:
  - Streamlit: "UI for the web application"
  - PyPDF2: "Extract text from PDF documents"
  - LangChain: "Text splitting, embeddings, chains"
  - FAISS: "Vector storage for document retrieval"
  - Google Generative AI API: "Embedding + Conversational model (Gemini Pro)"
  - dotenv: "Environment variable management"

structure:
  files:
    - main.py: "Main Streamlit application logic"
    - faiss_index/: "Directory to store FAISS vector DB"
    - .env: "Google API Key configuration"
    - requirements.txt: "Required Python packages"
    - README.md: "Documentation"

architecture:
  steps:
    - PDF Upload
    - Text Extraction using PyPDF2
    - Chunking using LangChain
    - Embeddings using Gemini
    - FAISS Vector Store
    - LangChain QA Chain + Gemini Pro
    - Streamlit Chat UI

setup:
  steps:
    - step: Clone the repository
      command: |
        git clone https://github.com/your-username/chat-with-pdf.git
        cd chat-with-pdf
    - step: Install dependencies
      command: pip install -r requirements.txt
    - step: Set up .env
      command: |
        echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
    - step: Run the app
      command: streamlit run main.py

usage:
  flow:
    - Upload PDFs
    - Click "Process PDFs"
    - Ask questions (comma-separated)
    - App performs vector search
    - Gemini Pro answers based on document context

session:
  features:
    - Save Session: "Stores current PDFs and questions"
    - Load Session: "Restores session to continue chat"

feedback: "Users can submit feedback on responses via the feedback text area in the UI."

future_enhancements:
  - OCR support for scanned PDFs
  - Choose between Gemini, OpenAI, Claude
  - User login and session persistence
  - Export chat history to PDF

contributing:
  instructions: >
    Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to improve.

license: "MIT License"

acknowledgements:
  - LangChain: "https://www.langchain.com/"
  - Google Generative AI: "https://ai.google.dev/"
  - Streamlit: "https://streamlit.io/"
