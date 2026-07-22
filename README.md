# Interview Helper

AI assistant for IT interviews.

## MVP scope

- near-real-time transcription through whisper.cpp;
- automatic and manual question confirmation;
- hybrid retrieval over an interview knowledge base;
- candidate-aware answer generation;
- streaming answer delivery with sources;
- replaceable STT, LLM, and embedding providers.

## Repository structure

```text
backend/                    FastAPI application and provider adapters
backend/ingestion/          Import, chunking, embeddings, and Hybrid RAG
backend/llm/                LLM contracts and OpenRouter client
backend/stt/                Streaming STT contracts and whisper.cpp HTTP client
backend/sources/            Knowledge source files
frontend/                   React + Vite interview interface
services/whispercpp/        Dedicated whisper-server CPU/CUDA images
docker-compose.yml          Portable CPU stack
docker-compose.gpu.yml      NVIDIA GPU override
```

## Architecture

```text
Browser microphone
  -> AudioWorklet: mono PCM16, 16 kHz
  -> FastAPI WebSocket
  -> rolling STT windows
  -> whispercpp container: whisper-server
  -> partial/final transcript events

Knowledge sources
  -> OpenRouter embeddings
  -> SQLite BM25 + vectors
  -> Hybrid RAG
  -> OpenRouter streaming LLM
```

The whisper.cpp model is loaded once by `whisper-server` and remains in the separate STT container. The backend sends temporary WAV windows to its internal `/inference` endpoint instead of launching a new process and loading the model for every window.

## Configuration

Copy the environment template and add an OpenRouter key:

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

Default remote providers:

```env
LLM_PROVIDER=openrouter
LLM_MODEL=~google/gemini-flash-latest
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
```

OpenRouter uses one API key and one OpenAI-compatible base URL for both `/chat/completions` and `/embeddings`. Set `OPENROUTER_HTTP_REFERER` when the application has a public URL; `OPENROUTER_APP_TITLE` controls the title shown in OpenRouter usage metadata.

## Start with CPU transcription

```bash
docker compose up --build
```

On first start, the whisper.cpp service downloads the selected GGML model into the persistent `whisper_models` volume. Later container rebuilds reuse the model.

Open:

- frontend: http://localhost:5173
- backend health: http://localhost:8000/api/v1/health
- backend OpenAPI: http://localhost:8000/docs
- whisper.cpp health on the local machine: http://localhost:8081/health

The whisper.cpp port is bound to `127.0.0.1` and is not exposed to the external network.

## Start with NVIDIA GPU transcription

Install the NVIDIA driver and NVIDIA Container Toolkit, then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The GPU override builds whisper.cpp with CUDA, requests all available GPUs for the STT container, and enables flash attention. The default CUDA image version is configurable through `CUDA_VERSION`.

## Transcription behavior

The browser captures microphone audio through an `AudioWorklet`, downsamples it to signed little-endian PCM16 mono at 16 kHz, and sends binary WebSocket frames to `/ws/interview/{session_id}`. JSON frames on the same socket control session start and stop.

Default rolling-window parameters:

- partial update every 3 seconds;
- final window every 12 seconds;
- 1 second overlap between final windows;
- automatic language detection;
- structured `stt_unavailable` and `stt_failed` errors;
- up to three frontend WebSocket reconnection attempts.

The backend checks `http://whispercpp:8080/health` when a session starts. The container exposes the official whisper.cpp `/health` and `/inference` endpoints and keeps the model resident between requests.

## Backend outside Docker

Start only the STT service:

```bash
docker compose up --build whispercpp
```

Then run the backend against the loopback port:

```bash
cd backend
poetry install
WHISPERCPP_BASE_URL=http://localhost:8081 \
poetry run uvicorn app.main:app --reload
```

## Build the knowledge index

The default index build uses OpenRouter embeddings:

```bash
cd backend
poetry run python -m ingestion build-index ./sources \
  --output ./data/index/knowledge.db
```

The same provider and model must be selected when querying. The index manifest stores the provider, model, dimensions, and vector count so incompatible runtime configuration fails explicitly.

For tests or completely offline development, use the deterministic hashing fallback:

```bash
poetry run python -m ingestion build-index ./sources \
  --output ./data/index/knowledge.db \
  --embedding-provider local-hash
```

Supported source formats are Markdown, HTML, and JSON. Frontmatter and JSON metadata are copied to chunks. Useful fields include `role`, `topic`, `level`, `language`, and `source_url`.

Search the hybrid index:

```bash
poetry run python -m ingestion search "reliable background jobs" \
  --index ./data/index/knowledge.db \
  --role "Python Developer"
```

Hybrid ranking uses semantic similarity, normalized BM25, and metadata relevance:

```text
final_score = 0.55 * semantic + 0.35 * bm25 + 0.10 * metadata
```

The backend exposes the same retrieval pipeline through:

```text
GET /api/v1/knowledge/search?q=reliable+jobs&limit=5&role=Python+Developer
```

## Frontend outside Docker

```bash
cd frontend
npm install
npm run dev
```

## Current milestone

The application now has browser audio capture, binary WebSocket transport, a dedicated persistent whisper.cpp service, partial/final transcription events, OpenRouter embeddings, Hybrid RAG, and a streaming OpenRouter LLM adapter. The next vertical slice consumes final transcript segments to detect and confirm interview questions before generating an answer.
