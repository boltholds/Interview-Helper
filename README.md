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
backend/            FastAPI application and domain contracts
backend/ingestion/  Offline import, chunking, embeddings, and hybrid retrieval
backend/stt/        Streaming STT contracts and whisper.cpp adapter
backend/sources/    Knowledge source files
frontend/           React + Vite interview interface
scripts/            Runtime setup scripts
runtime/            Local binaries and models, ignored by Git
docker-compose.yml
```

## Install whisper.cpp

The setup script pins whisper.cpp `v1.8.5`, builds `whisper-cli`, and downloads a multilingual GGML model into `runtime/whispercpp`:

```bash
bash scripts/setup-whispercpp.sh
```

The default model is `small`. Override it when needed:

```bash
WHISPER_CPP_MODEL=base bash scripts/setup-whispercpp.sh
```

For a CUDA build:

```bash
WHISPER_CPP_CUDA=1 bash scripts/setup-whispercpp.sh
```

The CUDA build requires a working CUDA toolkit. The default CPU build is sufficient for development and CI because tests use a fake command runner and do not download model weights.

## Local development

1. Install whisper.cpp as described above.

2. Copy environment variables:

```bash
cp .env.example .env
```

3. Start both applications:

```bash
docker compose up --build
```

4. Open:

- frontend: http://localhost:5173
- backend health: http://localhost:8000/api/v1/health
- OpenAPI: http://localhost:8000/docs

The browser captures microphone audio through an `AudioWorklet`, downsamples it to signed little-endian PCM16, mono, 16 kHz, and sends binary WebSocket frames to `/ws/interview/{session_id}`. JSON frames on the same socket control session start and stop.

The backend runs `whisper-cli` on rolling windows. Defaults:

- partial update every 3 seconds;
- final window every 12 seconds;
- 1 second overlap between final windows;
- language auto-detection;
- GPU and flash attention enabled when supported by the binary.

All values are configurable through `WHISPERCPP_*` environment variables. The backend validates the executable and model when `start_session` is received and returns a structured `stt_unavailable` error if the runtime is incomplete.

For backend development outside Docker, point the variables at the host runtime:

```bash
export WHISPERCPP_BINARY_PATH="$PWD/runtime/whispercpp/bin/whisper-cli"
export WHISPERCPP_MODEL_PATH="$PWD/runtime/whispercpp/models/ggml-small.bin"
cd backend
poetry install
poetry run uvicorn app.main:app --reload
```

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

## Frontend without Docker

```bash
cd frontend
npm install
npm run dev
```

## Current milestone

The application now has an end-to-end transcription path: browser microphone capture, PCM16 WebSocket transport, whisper.cpp rolling-window inference, partial/final transcript events, overlap-aware text merging, explicit runtime errors, and connection recovery. The next slice consumes final transcript segments to detect and confirm interview questions.
