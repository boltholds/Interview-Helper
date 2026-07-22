# Interview Helper

AI assistant for IT interviews.

## MVP scope

- near-real-time transcription;
- automatic and manual question confirmation;
- hybrid retrieval over an interview knowledge base;
- candidate-aware answer generation;
- streaming answer delivery with sources;
- replaceable STT, LLM, and embedding providers.

## Repository structure

```text
backend/            FastAPI application and domain contracts
backend/ingestion/  Offline import, chunking, and SQLite FTS5 indexing
backend/sources/    Knowledge source files
frontend/           React + Vite interview interface
docker-compose.yml
```

## Local development

1. Copy environment variables:

```bash
cp .env.example .env
```

2. Start both applications:

```bash
docker compose up --build
```

3. Open:

- frontend: http://localhost:5173
- backend health: http://localhost:8000/api/v1/health
- OpenAPI: http://localhost:8000/docs

## Build the knowledge index

From the `backend` directory:

```bash
poetry run python -m ingestion build-index ./sources \
  --output ./data/index/knowledge.db
```

Supported source formats are Markdown, HTML, and JSON. Markdown frontmatter and JSON metadata are copied to every generated chunk. Useful metadata fields include `role`, `topic`, `level`, `language`, and `source_url`.

Check the local BM25 search from the command line:

```bash
poetry run python -m ingestion search "reliable background jobs" \
  --index ./data/index/knowledge.db \
  --role "Python Developer"
```

The backend exposes the same index through:

```text
GET /api/v1/knowledge/search?q=reliable+jobs&limit=5&role=Python+Developer
```

Index creation is atomic: a temporary database is built first and replaces the active file only after a successful transaction.

## Backend without Docker

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload
```

## Frontend without Docker

```bash
cd frontend
npm install
npm run dev
```

## Current milestone

The repository now includes the first retrieval slice: source loading, text normalization, deterministic chunking, metadata preservation, a local SQLite FTS5/BM25 index, CLI commands, and a read-only search API. Semantic retrieval and score fusion are added in the next slice.
