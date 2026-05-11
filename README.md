<div align="center">

# DocuMind AI — RAG Document Q&A System

**Production-ready Retrieval-Augmented Generation system built with LangChain, FastAPI, ChromaDB, and Ollama**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-1C3C3C.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange.svg)](https://chromadb.com)
[![Ollama](https://img.shields.io/badge/Ollama-local-black.svg)](https://ollama.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow.svg)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 100% free and offline — no OpenAI API key or billing required.

</div>

---

## 📖 Overview

DocuMind AI lets you **upload documents** (PDF, DOCX, TXT) and **ask questions** about them using AI. The system retrieves relevant passages from your documents and uses a local LLM (via Ollama) to generate accurate, cited answers — everything runs on your own machine with no external API calls.

**Key advantages over cloud-based RAG:**
- ✅ Completely free — no API costs ever
- ✅ Fully offline — your documents never leave your machine
- ✅ No rate limits or quota errors
- ✅ Cites exact source chunks for every answer

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 Multi-format upload | PDF, DOCX, TXT, Markdown |
| 🔍 Semantic search | HuggingFace embeddings + cosine similarity |
| 🤖 Local LLM | Ollama (Mistral / LLaMA 3 / Gemma) — runs offline |
| 📌 Citations | Every answer shows source chunks + similarity scores |
| 💬 Chat history | Multi-turn conversations with context memory |
| ⚡ Streaming | Token-by-token SSE streaming for real-time UX |
| 🔄 Reranking | Cross-encoder reranking for improved precision |
| 🌙 Dark mode | Full dark/light theme support |
| 🐳 Docker | One-command deployment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  React + Tailwind UI                     │
│  DropZone → Upload  │  Chat Interface  │  Source Cards   │
└──────────────┬──────────────────────────────┬───────────┘
               │ REST / SSE                    │
┌──────────────▼──────────────────────────────▼───────────┐
│               FastAPI Backend (Python 3.11)              │
│                                                          │
│  DocumentProcessor          RAGPipeline                  │
│  ├── PyPDF / Docx2txt        ├── Embed query             │
│  ├── RecursiveChunker        ├── Similarity search       │
│  └── Embed + Store           ├── Rerank chunks           │
│                              ├── Augment prompt          │
│                              └── Stream LLM response     │
└──────────────┬──────────────────────────────┬───────────┘
               │                              │
┌──────────────▼──────────┐     ┌─────────────▼──────────┐
│  ChromaDB Vector Store  │     │   Ollama (local LLM)   │
│  HuggingFace Embeddings │     │   Mistral / LLaMA 3    │
│  all-MiniLM-L6-v2       │     │   runs on your machine │
└─────────────────────────┘     └────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Tailwind CSS, Zustand |
| Backend | FastAPI, Python 3.11, Uvicorn |
| RAG Framework | LangChain |
| Embeddings | HuggingFace — `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) / FAISS |
| LLM | Ollama — Mistral 7B (local, free, offline) |
| Reranking | sentence-transformers cross-encoder |
| Caching | Redis (optional) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed

### 1. Clone and configure
```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
cp .env.example .env
```

### 2. Start Ollama and pull a model
```bash
# Start Ollama (runs in background)
ollama serve

# Pull Mistral (4GB, best quality/speed balance)
ollama pull mistral

# OR pull a smaller model if low on RAM
ollama pull phi3        # 2GB, fast
ollama pull gemma:2b    # 1.5GB, lightest
```

### 3. Backend
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn backend.main:app --reload
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/api/v1/docs
```

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
# App running at http://localhost:3000
```

### 5. Docker (optional — runs everything together)
```bash
docker-compose up --build
```

---

## 🤖 Changing the Ollama Model

In `backend/services/rag_pipeline.py`:

```python
self.llm = Ollama(
    model="mistral",   # change to: llama3, gemma, phi3, llama2, etc.
    temperature=0.1,
)
```

| Model | Size | Best for |
|-------|------|----------|
| `mistral` | 4.1GB | Best quality/speed balance ✅ recommended |
| `llama3` | 4.7GB | Strong reasoning |
| `gemma:7b` | 5.0GB | Google's model, great at summarization |
| `phi3` | 2.3GB | Fast, low RAM usage |
| `gemma:2b` | 1.5GB | Lowest RAM, basic answers |

---

## 🔠 Embeddings

This project uses **HuggingFace sentence-transformers** — no API key needed, models download automatically and cache locally.

In `backend/services/vector_store.py`:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

| Model | Size | Notes |
|-------|------|-------|
| `all-MiniLM-L6-v2` | 80MB | Default — fast and accurate ✅ |
| `all-mpnet-base-v2` | 420MB | Higher quality, slower |
| `BAAI/bge-small-en-v1.5` | 130MB | Best for English RAG |

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check with component status |
| `POST` | `/api/v1/documents/upload` | Upload and index a document |
| `GET` | `/api/v1/documents/` | List all indexed documents |
| `GET` | `/api/v1/documents/{id}` | Get document metadata |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + embeddings |
| `POST` | `/api/v1/chat/query` | Non-streaming query |
| `POST` | `/api/v1/chat/stream` | SSE streaming query |

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "stream": false}'
```

---

## 📁 Folder Structure

```
rag-document-qa/
├── backend/
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py          # /chat/query and /chat/stream endpoints
│   │       ├── documents.py     # upload, list, delete endpoints
│   │       └── health.py        # health check endpoint
│   ├── core/
│   │   ├── config.py            # Pydantic settings + env vars
│   │   └── logging.py           # Structured JSON logging
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── services/
│   │   ├── document_processor.py  # Load → clean → chunk pipeline
│   │   ├── rag_pipeline.py        # Retrieval + Ollama generation
│   │   └── vector_store.py        # ChromaDB / FAISS abstraction
│   └── main.py                  # FastAPI app entry point
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Chat/            # MessageBubble, ChatInput, ChatWindow, SourceCard
│       │   ├── Sidebar/         # Sidebar, DocumentList
│       │   └── Upload/          # DropZone
│       ├── hooks/
│       │   ├── useChat.ts       # SSE streaming + abort logic
│       │   └── useDocuments.ts  # Upload with progress tracking
│       ├── store/
│       │   └── chatStore.ts     # Zustand state (messages, documents)
│       ├── styles/
│       │   └── globals.css      # Design tokens, animations, components
│       └── pages/
│           └── App.tsx          # Root layout + dark mode
│
├── tests/
│   ├── unit/
│   │   ├── test_document_processor.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_evaluation.py   # Precision@K, Recall@K, MRR metrics
│   └── integration/
│       └── test_api.py          # FastAPI endpoint tests
│
├── docker/
│   └── Dockerfile.backend
├── .github/
│   └── workflows/
│       └── ci.yml               # Lint, type-check, test, coverage
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Testing

```bash
# Run all tests with coverage report
pytest tests/ --cov=backend --cov-report=html -v

# Unit tests only
pytest tests/unit/ -v

# RAG evaluation metrics (Precision@K, Recall@K, MRR)
pytest tests/unit/test_evaluation.py -v

# Integration tests
pytest tests/integration/ -v
```

---

## 📊 RAG Evaluation Metrics

The project includes a full evaluation framework in `tests/unit/test_evaluation.py`:

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K retrieved chunks that are relevant |
| **Recall@K** | Fraction of all relevant chunks retrieved in top-K |
| **MRR** | Mean Reciprocal Rank — how early the first relevant chunk appears |
| **Latency P95** | 95th percentile retrieval latency in milliseconds |

---

## 🔬 Key Engineering Decisions

**Why Ollama over OpenAI?**
Ollama runs models locally — zero cost, zero latency from API calls, and your documents never leave your machine. Ideal for sensitive documents or projects without a budget.

**Why HuggingFace embeddings?**
`all-MiniLM-L6-v2` produces 384-dimensional embeddings, downloads once (~80MB), and runs fast on CPU. No API key, no usage costs, no rate limits.

**Why ChromaDB?**
Persistent on disk, supports metadata filtering, and has a clean Python API. Survives server restarts without rebuilding the index. FAISS is included as an alternative for high-throughput workloads.

**Why RecursiveCharacterTextSplitter?**
Splits on semantic boundaries (paragraphs → sentences → words) rather than fixed character counts, preserving meaning at chunk edges. The 200-character overlap prevents context loss between chunks.

**Why low temperature (0.1)?**
Makes the LLM deterministic and fact-focused — critical when grounding answers in specific documents. Higher temperature risks the model generating plausible-sounding but incorrect answers.

---

## 🚢 Deployment

### Backend → Railway
```bash
railway login
railway init
railway up
# Set environment variables in Railway dashboard
```

### Frontend → Vercel
```bash
cd frontend
vercel --prod
# Set VITE_API_URL=https://your-api.railway.app in Vercel dashboard
```

> Note: For deployment, Ollama must run as a separate service. Railway supports Docker deployments — use the included `docker-compose.yml`.

---

## 🛣️ Roadmap

- [ ] PostgreSQL for persistent document registry
- [ ] Multi-user authentication (Clerk / Auth0)
- [ ] Pinecone / Qdrant vector DB support
- [ ] OCR for scanned PDFs (Tesseract)
- [ ] RAG evaluation dashboard UI
- [ ] Analytics and usage tracking
- [ ] Multi-language document support

---

## 🤝 Contributing

Pull requests are welcome. For major changes please open an issue first.

```bash
# Setup pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## 📄 License

MIT © 2024 [Your Name]
