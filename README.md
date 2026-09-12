# aiQS - AI Document Question Answering System

`aiQS` is an AI-powered document exploration and Question-Answering system built using **Python**, **Streamlit**, and **Google Gemini API** (`google-genai`).

Users can upload documents (PDF, DOCX, TXT), which are automatically parsed, chunked, and embedded into a vector space. When asking questions, `aiQS` performs semantic similarity retrieval to find the most relevant context excerpts and generates answers strictly grounded in the document text, complete with page citations and transparent evidence cards.

---

## Key Features

- 📑 **Multi-Format Ingestion:** Supports **PDF**, **Word Documents (.docx)**, and **Plain Text/Markdown (.txt, .md)**.
- 🧩 **Semantic Chunking:** Preserves paragraph context, source filenames, and exact page numbers.
- ⚡ **NumPy Vector Store:** Fast, lightweight in-memory vector index with normalized cosine similarity ranking.
- 🎯 **Strict Grounding:** Zero hallucination prompting ensures answers are derived exclusively from document context.
- 🔍 **Evidence & Source Transparency:** Collapsible card view below every response showing the exact chunks, page numbers, and similarity scores retrieved.
- 🚀 **Interactive Streamlit Interface:** Modern chat UI with real-time streaming, sample prompts, and live document index stats.
- 🧪 **Pre-loaded Sample Document:** Includes an MCA Semester 1 syllabus to test the system with a single click.

---

## Project Structure

```
aiQS/
├── app.py                     # Streamlit web application & chat UI
├── document_loader.py         # Multi-format document parser & text chunker
├── vector_store.py            # NumPy in-memory vector database & cosine similarity
├── rag_engine.py              # Embedding & LLM grounding pipeline with Google GenAI
├── test_rag.py                # Automated unit & integration tests
├── requirements.txt           # Project dependencies
├── sample_documents/
│   └── MCA_Semester_1_Syllabus.txt  # Ready-to-test syllabus document
├── .env.example               # Environment variable configuration template
└── README.md                  # Project documentation
```

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.10+ installed
- Google Gemini API key ([Get a free key from Google AI Studio](https://aistudio.google.com/))

### 2. Setup Virtual Environment
```bash
# Clone or open the aiQS directory
cd aiQS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
You can either:
1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`:
   ```bash
   cp .env.example .env
   ```
   Add your key:
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```
2. **Or** simply enter your API key directly into the secure sidebar input field in the Streamlit web application.

### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Running Tests
Run the test suite to verify document loading, parsing, chunking, and similarity search:
```bash
python test_rag.py
```
