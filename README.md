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
backend/ingestion/  Offline import, chunking, embeddings, and hybrid retrieval
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

The default `local-hash` provider is deterministic, requires no model download, and is intended for development and offline tests. To build a production-quality semantic index through OpenAI embeddings:

```bash
export OPENAI_API_KEY="..."
poetry run python -m ingestion build-index ./sources \
  --output ./data/index/knowledge.db \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small
```

The same provider and model must be selected when querying the index. The index manifest stores the provider, model, dimensions, and vector count so incompatible runtime configuration fails explicitly instead of silently degrading retrieval.

Supported source formats are Markdown, HTML, and JSON. Markdown frontmatter and JSON metadata are copied to every generated chunk. Useful metadata fields include `role`, `topic`, `level`, `language`, and `source_url`.

Search the hybrid index from the command line:

```bash
poetry run python -m ingestion search "reliable background jobs" \
  --index ./data/index/knowledge.db \
  --role "Python Developer"
```

Hybrid ranking uses semantic similarity, normalized BM25, and metadata relevance:

```text
final_score = 0.55 * semantic + 0.35 * bm25 + 0.10 * metadata
```

Candidates from both retrieval layers are merged and deduplicated by content hash. Exact filters are available for `role`, `topic`, `level`, and `language`.

The backend exposes the same retrieval pipeline through:

```text
GET /api/v1/knowledge/search?q=reliable+jobs&limit=5&role=Python+Developer
```

Each result includes raw and normalized lexical/semantic scores plus the final fused score. Index creation remains atomic: a temporary database is built first and replaces the active file only after a successful transaction.

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

The retrieval vertical slice now covers source loading, deterministic chunking, metadata preservation, SQLite FTS5/BM25, local vector persistence, cosine semantic search, score fusion, filtering, deduplication, CLI commands, and a read-only Hybrid RAG API. A neural reranker and evaluation dataset remain separate follow-up slices.
