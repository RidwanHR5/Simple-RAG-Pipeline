# Advanced RAG Q&A System

A sophisticated Retrieval-Augmented Generation (RAG) question-answering system built with Streamlit, LangChain, and Google Gemini. This application allows you to chat with your documents by creating a knowledge base from various sources and asking questions in natural language.

## 🚀 Features

- **Multiple Input Formats**: Support for PDF, DOCX, TXT files, web links, and plain text
- **Interactive Chat Interface**: Streamlit-based UI for seamless document interaction
- **History-Aware Retrieval**: Contextual question reformulation based on chat history
- **Advanced RAG Pipeline**: Uses FAISS vector store with HuggingFace embeddings
- **Google Gemini Integration**: Powered by Gemini 3 Pro for high-quality responses
- **Source Attribution**: View the source documents used for each answer
- **Validation System**: Built-in RAGAS evaluation framework for testing

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini)
- HuggingFace API Token (for embeddings)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Simple-RAG-Pipeline
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Add your documents**
   - Use the sidebar to select input type (Link, PDF, Text, DOCX, or TXT)
   - Upload files or enter content
   - Click "Process Data" to create the knowledge base

3. **Start chatting**
   - Once the knowledge base is created, you can ask questions in the chat interface
   - The system will retrieve relevant context and generate answers
   - View source documents by expanding the "📚 View Sources" section

### Running Validation

To evaluate the RAG system using the Amnesty QA dataset:

```bash
python run_validation.py
```

This will:
- Load the test dataset
- Process all contexts into a knowledge base
- Run inference on all test questions
- Calculate RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
- Save results to `validation_runs/` directory with timestamp

## 📁 Project Structure

```
Simple-RAG-Pipeline/
├── app.py                 # Main Streamlit application
├── run_validation.py      # RAGAS validation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
└── validation_runs/      # Validation results (auto-generated)
    └── YYYYMMDD_HHMMSS/
        ├── detailed_results.csv
        └── summary_scores.json
```

## ⚙️ Configuration

Key configuration parameters in `app.py`:

- **Chunking**: `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 200`
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **LLM Model**: `gemini-3-pro-preview`
- **Retrieval**: Top K = 5 documents

You can modify these values in the configuration section of `app.py` to suit your needs.

## 🔧 Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: RAG pipeline orchestration
- **Google Gemini**: Large language model
- **HuggingFace**: Embeddings and transformers
- **FAISS**: Vector similarity search
- **RAGAS**: RAG evaluation framework
- **PyPDF2**: PDF processing
- **python-docx**: DOCX file processing

## 📝 Supported File Types

- **PDF**: `.pdf` files
- **Word Documents**: `.docx`, `.doc` files
- **Text Files**: `.txt` files
- **Web Links**: URLs (up to 20 links)
- **Plain Text**: Direct text input

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- LangChain community for the excellent RAG framework
- Google for Gemini API
- HuggingFace for embeddings and models
