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
backend/   FastAPI application and domain contracts
frontend/  React + Vite interview interface
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

The first milestone establishes executable frontend/backend skeletons and stable transport/domain contracts. STT, question detection, retrieval, and generation are implemented in subsequent slices.
