# Contributing to RAG Document Q&A

## Development Setup

```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI key
```

## Running Tests

```bash
pytest tests/ -v         # All tests
pytest tests/unit/ -v    # Unit tests only (fast, no API)
```

## Code Style

We use `ruff` for linting:
```bash
ruff check backend/
ruff format backend/
```

## Pull Request Process

1. Branch from `develop` (not `main`)
2. Write tests for new features
3. Ensure all tests pass
4. Update README if adding new features
5. PR description should explain the change and why

## Architecture Decisions

See `docs/` for ADRs (Architecture Decision Records) explaining key choices:
- Why FAISS over Pinecone
- Why LangChain over raw OpenAI SDK
- Chunking strategy rationale
