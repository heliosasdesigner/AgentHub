# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

### Environment Setup
- Create virtual environment: `uv venv .venv && source .venv/bin/activate`
- Install dependencies: `uv pip install -r requirements.txt`
- Copy environment config: `cp .env.example .env` (then configure DB_URL, LLM keys, Neo4j credentials)

### Running the Application
- Backend: `cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 11111`
- Frontend: `cd frontend && python app_gradio.py` (uses BACKEND_URL env var, defaults to http://localhost:11111)
- Docker: `docker compose up --build` (starts backend on port 11111 and Neo4j)

### Testing
- Run all tests: `.venv/bin/pytest`
- Tests use isolated SQLite database (DB_URL set automatically in test setup)
- Tests mock LLM and external service calls

## High-Level Architecture

### System Design
This is an AI agent experiment platform with pluggable agent architectures. The backend exposes a unified `/chat` API that routes to different agent implementations (RAG-Faiss, GraphRAG-Neo4j, Cognee), each with its own storage backend.

**Key principle**: Frontend communicates only with the backend API. Each agent owns its internal storage (vector DB, graph DB, etc.). Conversations and messages are stored centrally in a shared SQLite/Postgres database.

### Core Flow
1. Frontend (Gradio) sends chat requests to `/chat` with `agent_id`, `session_id`, and messages
2. Backend validates agent, ensures conversation exists, logs user message
3. Agent registry routes to specific agent runner (rag-faiss, graphrag-neo4j, cognee)
4. Agent processes request, returns response
5. Backend logs assistant message and run metrics
6. Frontend receives and displays response

### Module Structure
- `backend/app/main.py`: FastAPI entrypoint with lifespan that initializes DB and agents
- `backend/app/api/`: HTTP route handlers
  - `routes_chat.py`: `/chat` endpoint (main interaction point)
  - `routes_agents.py`: `/agents` endpoint (list available agents)
  - `routes_sessions.py`: session management endpoints
- `backend/app/agents/`: Agent implementations
  - `base.py`: Core types (Message, AgentInput, AgentOutput)
  - `registry.py`: AGENT_REGISTRY dict mapping agent_id to callable runners
  - `rag_faiss/`, `graphrag_neo4j/`, `cognee_agent/`: Individual agent implementations
  - Each agent has: `builder.py` (constructs agent runner), `retriever.py`/`ingest.py` (storage logic)
- `backend/app/persistence/`: Database layer
  - `db.py`: SQLAlchemy session management
  - `models.py`: ORM models (AgentRecord, ConversationRecord, MessageRecord, RunRecord)
- `backend/app/services/`: Helper services
  - `history_service.py`: High-level conversation/message operations
  - `llm_router.py`: Routes between OpenAI/vLLM endpoints based on config
- `backend/app/graph/`: LangGraph utilities
  - `state.py`: MessagesState for agent state management
  - `utils.py`: Helper functions for message manipulation
- `frontend/app_gradio.py`: Gradio chat UI that calls backend `/chat`

### Agent System
Agents are registered at startup in `registry.py:init_agents()`. Each agent:
- Has metadata (id, name, description) stored in AGENT_METADATA list
- Implements an async callable: `AgentInput -> AgentOutput`
- Can use LangGraph for orchestration or simple async functions
- Maintains its own retrieval/storage backend (Faiss index, Neo4j graph, Cognee store)

### Database Schema
- `agents`: Agent metadata (id, name, description, config)
- `conversations`: Chat sessions (id=session_id, agent_id, title, user_id)
- `messages`: Individual messages (conversation_id, role, content, created_at)
- `runs`: Execution logs (conversation_id, agent_id, started_at, finished_at, success, metrics_json)

Conversations are tied to a single agent_id. Attempting to use different agents with same session_id will error.

## Configuration and Settings

### Environment Variables
See `.env.example` for full list. Key variables:
- `DB_URL`: Database connection (default: `sqlite:///./agent_lab.db`)
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `VLLM_BASE_URL`: vLLM endpoint (default: `http://localhost:8000/v1`)
- `VLLM_API_KEY`: vLLM authentication key
- `VLLM_MODEL`: Model name for vLLM (default: `meta-llama/Meta-Llama-3-8B-Instruct`)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection for graphrag-neo4j agent
- `COGNEE_DATA_DIR`: Data directory for Cognee agent

Settings are loaded via pydantic-settings in `backend/app/config.py`. The `Settings` class reads from `.env` file.

### LLM Routing
`services/llm_router.py` provides `resolve_llm_endpoint()` which selects between:
1. Explicitly provided model/base_url/api_key (from request config)
2. OpenAI (if OPENAI_API_KEY is set)
3. vLLM (fallback, using VLLM_* env vars)

## Adding New Agents

1. Create agent directory under `backend/app/agents/your_agent/`
2. Implement `builder.py` with a `build_your_agent()` function returning `AgentRunner`
3. Add agent metadata to `AGENT_METADATA` list in `registry.py`
4. Register builder in `init_agents()` function: `AGENT_REGISTRY["your-id"] = build_your_agent()`
5. Implement retrieval/ingest logic in separate modules as needed
6. Agent must accept `AgentInput` and return `AgentOutput`

## Testing Practices

- Test files live in `tests/` directory
- Tests set `DB_URL` to isolated SQLite file before importing app modules
- Name tests after behavior: `test_rag_returns_stub_reply`, not `test_run_agent`
- Mock external dependencies (LLM calls, Neo4j, Faiss) to keep tests fast and isolated
- Use `session_scope()` context manager pattern for database session management in tests
- Setup/teardown module functions handle test database creation and cleanup

## Docker and Neo4j

- `docker-compose.yml` defines backend service and Neo4j service
- Backend depends on Neo4j and mounts current directory as `/app`
- Neo4j uses official `neo4j:5.12` image with authentication from env vars
- Neo4j browser: http://localhost:7474
- Neo4j bolt: bolt://localhost:7687
- Volumes persist Neo4j data and logs

## Additional Notes

- Python 3.11+ required
- Uses FastAPI with async/await throughout
- Type hints are used consistently across the codebase
- Agent implementations are currently stubs returning placeholder responses
- Faiss is an optional dependency (install separately if enabling rag-faiss)
- The platform is designed for experimentation and comparison of different agent architectures
- See `docs/ai-agent-implementation.md` for detailed implementation guide and architecture diagrams
