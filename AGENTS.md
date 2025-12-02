# Repository Guidelines

## Project Structure & Module Organization
- `backend/app`: FastAPI application code. Key areas: `api/` (HTTP routes), `agents/` (agent runners and registry), `persistence/` (SQLAlchemy models and DB wiring), `services/` (helpers), `graph/` (LangGraph utilities).
- `frontend/app_gradio.py`: Gradio chat UI that calls the backend `/chat`.
- `docs/`: Reference guides, including `ai-agent-implementation.md`.
- `requirements.txt`, `pyproject.toml`: Dependency definitions; `Dockerfile` and `docker-compose.yml` for containerized runs.

## Build, Test, and Development Commands
- Create env: `uv venv .venv && source .venv/bin/activate`.
- Install deps (includes Hugging Face CLI extras pinned for Gradio compatibility): `uv pip install -r requirements.txt`. If you upgraded `huggingface-hub`, reinstall to 0.23.5 to keep `HfFolder` available.
- LLM runners (OpenAI-compatible): prefer LM Studio server on `http://localhost:1234/v1` (set `LMSTUDIO_*` envs). Optional: point to an external vLLM deployment via `VLLM_*`. Router order: explicit request → OpenAI (if keyed) → LM Studio → vLLM.
- Run backend: `cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 11111`.
- Run frontend: `cd frontend && python app_gradio.py` (uses `BACKEND_URL` env; defaults to http://localhost:11111).
- Docker (Neo4j only): `docker compose up -d` (starts Neo4j on 7474/7687). Run the backend locally and point `NEO4J_URI=bolt://localhost:7687`.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation. Favor type hints throughout.
- Keep code ASCII unless existing file requires otherwise.
- Prefer descriptive function and module names (e.g., `build_rag_faiss_agent`, `history_service`).
- Add comments sparingly—only to clarify non-obvious logic or tricky flows.

## Testing Guidelines
- Tests live under `tests/`; keep them fast and isolated from external services. Mock LLM and DB calls where possible.
- Run suite: `.venv/bin/pytest` (tests set `DB_URL` to an isolated SQLite file).
- Name tests after behavior, not method names (e.g., `test_rag_returns_stub_reply`).

## Documentation & LLM Stack Updates
- When changing the LLM/embedding stack (now default: vLLM `openai/gpt-oss-20b` + `BAAI/bge-m3`), consolidate updates across docs (`docs/ai-agent-implementation.md`) and env examples (`.env.example`).
- Keep AGENTS.md in sync with any new steps or tooling expectations.
- Before merging, run the full test suite (`.venv/bin/pytest`) and ensure it passes 100%.

## Commit & Pull Request Guidelines
- Keep commits scoped to a single concern; use clear, present-tense subjects (e.g., “Add Neo4j graph store stub”).
- PRs should summarize changes, list testing performed, and link issues or tasks. Include screenshots or logs only when UI or behavior changes.

## Security & Configuration Tips
- Copy `.env.example` to `.env` and avoid committing secrets. Configure DB (`DB_URL`), LLM keys, and Neo4j credentials there.
- Default SQLite is fine for local dev; set Postgres/Neo4j env vars for production-like runs.

## Agent-Specific Notes
- Registry lives in `backend/app/agents/registry.py`; add new agents there and supply metadata.
- Each agent keeps its own storage logic (`rag_faiss`, `graphrag_neo4j`, `cognee_agent`). Wire real retrievers/ingest pipelines in their `builder.py` and `ingest.py` stubs. Faiss is optional—install it separately if you enable that agent.
