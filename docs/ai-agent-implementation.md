# AI Agent Experiment Platform – Implementation Guide (v0.1)

## 0. Purpose & Scope

Local AI agent lab where you can:

- Test different agent architectures behind a single API:
  - graphrag-neo4j
  - rag-faiss
  - cognee
  - ...future options
- Keep a shared conversation DB (SQLite or Postgres).
- Let the frontend (Gradio) switch agents per session without changing its code.

Key design rule:

- Frontend talks only to a single Agent Experiment Backend.
- Each agent architecture owns its own internal stores (graph DB, vector DB, Cognee store, etc.).

## 1. High-Level Architecture

### 1.1 System overview

```
flowchart LR
  subgraph Frontend
    FE[Gradio UI or Web App]
  end

  subgraph Backend
    API[HTTP API and Session Router]
    AR[Agent Registry and Config Store]
    ENG[Agent Engine]
  end

  subgraph ConversationDB
    CONV[(Conversations)]
    MSG[(Messages)]
    RUN[(Runs and Experiments)]
  end

  subgraph AgentDeps
    A1[(Neo4j GraphRAG)]
    A2[(Faiss Index)]
    A3[(Cognee Store)]
    AX[(Other Stores)]
  end

  FE -->|chat with agent id and session id| API
  API --> AR
  API --> ENG
  ENG --> CONV
  ENG --> MSG
  ENG --> RUN

  ENG -->|Option 1| A1
  ENG -->|Option 2| A2
  ENG -->|Option 3| A3
  ENG -->|Option N| AX
```

### 1.2 Core concepts

- **Agent**: specific architecture + config bundle, e.g. graphrag-neo4j (LangGraph + Neo4j GraphRAG), rag-faiss (LangChain RAG + Faiss index), cognee (Cognee-based pipeline).
- **Session / Conversation**: logical thread of messages linked to agent_id + session_id (UUID) stored in conversations + messages tables.
- **Agent Registry**: central map agent_id -> callable (compiled LangGraph or equivalent) that knows how to handle a `/chat` request.

## 2. Tech Stack

### 2.1 Backend

- Language: Python 3.11+
- Web framework: FastAPI
- Orchestrator: LangGraph
- LLM tools: LangChain
- Database: default SQLite for local dev, optional Postgres in production
- Per-agent deps:
  - graphrag-neo4j: Neo4j server
  - rag-faiss: Faiss (in-process index)
  - cognee: Cognee library + its storage

### 2.2 Frontend

- Gradio app (Python)
- Renders chat UI
- Lets user choose agent_id
- Calls backend via HTTP

## 3. Repository Layout

```
ai-agent-lab/
  backend/
    app/
      __init__.py
      main.py              # FastAPI entrypoint
      config.py            # env + settings
      api/
        __init__.py
        routes_chat.py     # /chat endpoint
        routes_agents.py   # /agents, /sessions
      agents/
        __init__.py
        base.py            # Agent interface / types
        registry.py        # AGENT_REGISTRY
        rag_faiss/
          __init__.py
          builder.py       # build_rag_faiss_agent()
          retriever.py
        graphrag_neo4j/
          __init__.py
          builder.py       # build_graphrag_neo4j_agent()
          graph_store.py
          ingest.py        # CLI for building graph
        cognee_agent/
          __init__.py
          builder.py       # build_cognee_agent()
      graph/
        state.py           # LangGraph state (MessagesState, etc.)
        utils.py
      persistence/
        db.py              # DB connection (SQLAlchemy)
        models.py          # ORM models for conversations/messages/runs
      services/
        llm_router.py      # choose OpenAI vs local runners (LM Studio first, vLLM optional)
        history_service.py # high-level history ops

  frontend/
    app_gradio.py          # Gradio UI, calls backend /chat

  docs/
    ai-agent-implementation.md

  .env.example
  requirements.txt
  docker-compose.yml       # optional: neo4j, backend, etc.
```

## 4. Configuration

Use `.env` + pydantic settings or similar.

Example `.env`:

```
# Core
APP_ENV=dev
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# DB
DB_URL=sqlite:///./agent_lab.db
# or:
# DB_URL=postgresql+psycopg2://user:password@localhost:5432/agent_lab

# LLMs
OPENAI_API_KEY=sk-...
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=
LMSTUDIO_MODEL=
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-secret-key
VLLM_MODEL=openai/gpt-oss-20b

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpassword

# Cognee, etc. (if needed)
COGNEE_DATA_DIR=./cognee_data

# Embeddings
EMBED_MODEL_ID=BAAI/bge-m3
```

### 4.1 Local Neo4j with Docker (backend runs locally)

- From repo root: `docker compose up -d` (starts Neo4j only). Bolt: `bolt://localhost:7687`; Browser: `http://localhost:7474`.
- In `.env`, set `NEO4J_URI=bolt://localhost:7687` and `NEO4J_USER`/`NEO4J_PASSWORD` (defaults: `neo4j` / `neo4jpassword`).
- Run the FastAPI backend locally (uvicorn) and it will connect to the Neo4j container via those envs.

`backend/app/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: str = "dev"
    db_url: str
    openai_api_key: str | None = None
    lmstudio_base_url: str | None = None
    lmstudio_api_key: str | None = None
    lmstudio_model: str | None = None
    vllm_base_url: str | None = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_model: str = "openai/gpt-oss-20b"
    embed_model_id: str = "BAAI/bge-m3"

    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
```

## 5. Conversation DB Schema

Store conversations once, regardless of which agent generated them.

### 5.1 Tables

SQL-ish definition:

```
CREATE TABLE agents (
  id TEXT PRIMARY KEY,          -- 'graphrag-neo4j'
  name TEXT NOT NULL,
  description TEXT,
  config_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
  id TEXT PRIMARY KEY,          -- session_id (UUID)
  agent_id TEXT NOT NULL,
  title TEXT,
  user_id TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  meta_json TEXT,
  FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,           -- 'user' | 'assistant' | 'system' | 'tool'
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  meta_json TEXT,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  finished_at TIMESTAMP,
  success BOOLEAN,
  metrics_json TEXT,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id),
  FOREIGN KEY (agent_id) REFERENCES agents(id)
);
```

Use SQLAlchemy models for actual implementation.

## 6. Backend: Agent Registry & API Contract

### 6.1 Agent I/O types

`backend/app/agents/base.py`:

```python
from typing import Any, List, Dict, Literal
from pydantic import BaseModel

Role = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    role: Role
    content: str

class AgentInput(BaseModel):
    agent_id: str
    session_id: str
    messages: List[Message]
    config: Dict[str, Any] = {}

class AgentOutput(BaseModel):
    reply: Message
    metadata: Dict[str, Any] = {}
```

Each agent implementation exposes something like:

```python
from .base import AgentInput, AgentOutput

async def run_agent(input: AgentInput) -> AgentOutput:
    ...
```

or a LangGraph CompiledGraph with a wrapper.

### 6.2 Agent Registry

`backend/app/agents/registry.py`:

```python
from typing import Callable, Dict
from .base import AgentInput, AgentOutput
from .rag_faiss.builder import build_rag_faiss_agent
from .graphrag_neo4j.builder import build_graphrag_neo4j_agent
from .cognee_agent.builder import build_cognee_agent

AgentRunner = Callable[[AgentInput], AgentOutput]

AGENT_REGISTRY: Dict[str, AgentRunner] = {}

def init_agents() -> None:
    AGENT_REGISTRY["rag-faiss"] = build_rag_faiss_agent()
    AGENT_REGISTRY["graphrag-neo4j"] = build_graphrag_neo4j_agent()
    AGENT_REGISTRY["cognee"] = build_cognee_agent()
```

Each `build_*` returns a callable that internally uses LangGraph / LangChain.

## 7. Backend API

### 7.1 `/agents` (GET)

- Purpose: list available agents for the UI.
- Response example:

```json
[
  {
    "id": "rag-faiss",
    "name": "RAG (Faiss)",
    "description": "Classic vector RAG using Faiss index"
  },
  {
    "id": "graphrag-neo4j",
    "name": "GraphRAG (Neo4j)",
    "description": "Graph-based RAG over Neo4j property graph"
  },
  {
    "id": "cognee",
    "name": "Cognee GraphRAG",
    "description": "GraphRAG using Cognee stack"
  }
]
```

### 7.2 `/chat` (POST)

Request body:

```json
{
  "agent_id": "graphrag-neo4j",
  "session_id": "abc123",
  "messages": [
    {"role": "user", "content": "Explain relationship between X and Y"}
  ],
  "config": {
    "model": "openai/gpt-oss-20b",
    "base_url": "http://localhost:8000/v1",
    "api_key": "local-secret-key",
    "temperature": 0.3
  }
}
```

Backend flow:

- Validate `agent_id` exists in registry.
- Ensure `session_id` exists in conversations table (create if new).
- Insert user message into messages.
- Call corresponding agent runner.
- Log metrics in runs.
- Insert assistant message into messages.
- Return assistant message to frontend.

Response example:

```json
{
  "reply": {
    "role": "assistant",
    "content": "Here is how X connects to Y..."
  },
  "metadata": {
    "agent_id": "graphrag-neo4j",
    "tokens": {"prompt": 512, "completion": 240},
    "latency_ms": 1340
  }
}
```

### 7.3 `/sessions/{session_id}` (GET)

Fetch messages for a conversation so the frontend can reload history.

## 8. Frontend (Gradio)

`frontend/app_gradio.py` (conceptual):

```python
import gradio as gr
import requests
import uuid

BACKEND_URL = "http://localhost:8000"

def get_agents():
    return requests.get(f"{BACKEND_URL}/agents").json()

AGENTS = get_agents()
AGENT_CHOICES = [a["id"] for a in AGENTS]

def chat_fn(history, agent_id, session_id):
    if not session_id:
        session_id = str(uuid.uuid4())

    messages = [{"role": "user", "content": history[-1][0]}]

    payload = {
        "agent_id": agent_id,
        "session_id": session_id,
        "messages": messages,
        "config": {}
    }
    resp = requests.post(f"{BACKEND_URL}/chat", json=payload).json()
    assistant_reply = resp["reply"]["content"]
    history[-1][1] = assistant_reply
    return history, session_id

with gr.Blocks() as demo:
    session_state = gr.State()
    agent_id = gr.Dropdown(choices=AGENT_CHOICES, value=AGENT_CHOICES[0], label="Agent")
    chat = gr.Chatbot()
    msg = gr.Textbox()

    def on_submit(user_msg, history, agent_id, session_id):
        history = history + [[user_msg, None]]
        return chat_fn(history, agent_id, session_id)

    msg.submit(on_submit, [msg, chat, agent_id, session_state], [chat, session_state])

demo.launch()
```

## 9. Implementing Individual Agents

### 9.1 rag-faiss (Option 1, simplest baseline)

- Goal: classic vector RAG to serve as baseline architecture.
- Implementation sketch:
  - Build Faiss index from local documents (offline script).
  - Wrap as LangChain FAISS vectorstore + retriever.
  - Agent graph: router node (always → RAG for now), retrieval node (top-k chunks), answer node (LLM over context + question).
- `backend/app/agents/rag_faiss/builder.py` returns a function:

```python
def build_rag_faiss_agent() -> AgentRunner:
    retriever = load_faiss_retriever(...)

    async def run_agent(input: AgentInput) -> AgentOutput:
        # Use history if needed; here we just take last user msg
        question = input.messages[-1].content
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)
        answer = call_llm(question, context, input.config)
        return AgentOutput(
            reply=Message(role="assistant", content=answer),
            metadata={"num_docs": len(docs)}
        )

    return run_agent
```

### 9.2 graphrag-neo4j (Option 2, graph-based)

- Goal: GraphRAG architecture using Neo4j.
- Components: `graph_store.py` (Neo4j driver + helper functions), `ingest.py` (CLI to build graph), `builder.py` builds a LangGraph that classifies intent, generates Cypher for graph retrieval, summarises subgraph, calls LLM with graph-derived context.

### 9.3 cognee (Option 3, Cognee-powered)

- Goal: integrate Cognee’s internal graph/RAG pipeline as an alternative architecture.
- Implementation: use Cognee ingestion + retrieval APIs, wrap Cognee retriever in a simple agent runner or small LangGraph, keep all Cognee-specific logic inside `agents/cognee_agent`.

## 10. Running Locally (Dev Workflow)

- Install Python deps with `uv` (faster pip/venv)

```
cd ai-agent-lab
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -U "huggingface-hub[cli]"  # install huggingface-cli for token login/upload
```

- Copy and edit `.env`

```
cp .env.example .env
# set DB_URL, OpenAI key, Neo4j credentials, etc.
```

- Choose an LLM runner (OpenAI-compatible HTTP):
  - LM Studio (default for local dev): start LM Studio’s server (default base URL `http://localhost:1234/v1`). Set `LMSTUDIO_BASE_URL`/`LMSTUDIO_MODEL` and leave `OPENAI_API_KEY` empty.
  - vLLM (external/other project): point `VLLM_BASE_URL`/`VLLM_MODEL` at a running vLLM instance; this repo does not start it.
  - Router order: explicit request → OpenAI key (if set) → LM Studio → vLLM fallback.

- Run per-agent infrastructure (optional)
  - Start Neo4j (Docker or system). If using the bundled Compose file (Neo4j only): from repo root run `docker compose up -d` to expose Bolt on `bolt://localhost:7687` and HTTP on `http://localhost:7474`. In `.env`, set `NEO4J_URI=bolt://localhost:7687` and matching `NEO4J_USER`/`NEO4J_PASSWORD` (defaults: `neo4j` / `neo4jpassword`).
  - Build Faiss index via `python -m backend.app.agents.rag_faiss.ingest`.
  - Ingest graph for Neo4j via `python -m backend.app.agents.graphrag_neo4j.ingest`.

- Start backend

```
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 11111
```

- Start frontend

```
cd frontend
python app_gradio.py
```

- Open browser to Gradio URL (usually http://127.0.0.1:7860), pick an agent from the dropdown, chat and compare outputs. Backend default: http://127.0.0.1:11111.

## 11. Testing & Debugging

- Unit tests: for each agent module (`rag_faiss`, `graphrag_neo4j`, `cognee_agent`), create `tests/test_*.py`, mock LLM calls to avoid API usage.
- DB checks: inspect `agent_lab.db` with sqlite3 or a GUI. Verify conversations created, messages inserted, runs logged.
- Agent-specific debugging:
  - rag-faiss: validate embeddings and nearest neighbours.
  - graphrag-neo4j: log generated Cypher and inspect Neo4j browser.
  - cognee: inspect Cognee’s graph UI / logs.
## 12. vLLM + NVIDIA GPU via Docker (Ubuntu) – gpt-oss-20b + BGE-M3

LM Studio (desktop) is another local runner: start its server (default `http://localhost:1234/v1`), load a model, and set `LMSTUDIO_BASE_URL`/`LMSTUDIO_MODEL` in `.env`. The steps below focus on the vLLM container path.

Locked defaults:
- Reasoning LLM: `openai/gpt-oss-20b` served by vLLM
- Embeddings: `BAAI/bge-m3` via FlagEmbedding inside the backend

### 12.0 Goal

1. Run a Dockerised vLLM server (with NVIDIA GPU) serving `openai/gpt-oss-20b` over an OpenAI-compatible HTTP API (`/v1/...`).
2. Use BGE-M3 embeddings via the FlagEmbedding Python library from the backend.

### 12.1 Architecture

```mermaid
flowchart LR
  subgraph Host["Ubuntu + NVIDIA GPU"]
    D[Docker Engine]
    GPU[(NVIDIA GPU)]
  end

  subgraph vLLMContainer["Docker: vLLM + gpt-oss-20b"]
    S[OpenAI-compatible API<br/>http://mlbox.local:8000/v1]
  end

  subgraph Backend["Agent Backend (FastAPI / LangGraph / etc.)"]
    E[BAAI/bge-m3<br/>FlagEmbedding]
    A[Agents, RAG, GraphRAG]
  end

  D --> vLLMContainer
  vLLMContainer --> GPU
  Backend -->|Chat / Tools| vLLMContainer
  Backend -->|Embeddings| E
```

- All reasoning/generation: `gpt-oss-20b` via vLLM.
- All embeddings: `BAAI/bge-m3` via FlagEmbedding in-process.

### 12.2 Models

- gpt-oss-20b: `openai/gpt-oss-20b` (Hugging Face). MoE with ~3.6B active params, designed for ~16GB VRAM. Uses MXFP4; requires the gpt-oss vLLM wheel (`vllm==0.10.1+gptoss`).
- BGE-M3: `BAAI/bge-m3` (Hugging Face). XLM-R based; multilingual; up to ~8k tokens; supports dense, sparse, and ColBERT-style multi-vector in one encoder via `BGEM3FlagModel`.

### 12.3 Prerequisites

- Ubuntu with NVIDIA GPU and drivers (`nvidia-smi` works).
- Docker Engine.
- NVIDIA Container Toolkit (for `--gpus all`).
- Hugging Face account and token with access to `openai/gpt-oss-20b` and `BAAI/bge-m3`.

### 12.4 Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 12.5 Prepare cache directories

Persist HF cache on host:

```bash
sudo mkdir -p /opt/vllm/hf-cache
sudo chown -R $USER:$USER /opt/vllm
```

### 12.6 Build a gpt-oss-ready vLLM image

Create `Dockerfile.vllm-gpt-oss`:

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip install --upgrade pip && \
    pip install --pre "vllm==0.10.1+gptoss" \
      --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
      --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
      --index-strategy unsafe-best-match

ENV HUGGING_FACE_HUB_TOKEN=""
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

EXPOSE 8000

CMD ["vllm", "serve", "openai/gpt-oss-20b", "--host", "0.0.0.0", "--port", "8000"]
```

Build it:

```bash
docker build -f Dockerfile.vllm-gpt-oss -t local/vllm-gpt-oss:20b .
```

### 12.7 Run the container

```bash
export HF_TOKEN=hf_xxx_your_token
export VLLM_API_KEY=local-secret-key

docker run -d \
  --name vllm-gpt-oss-20b \
  --gpus all \
  --ipc=host \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  -e VLLM_API_KEY=${VLLM_API_KEY} \
  -v /opt/vllm/hf-cache:/root/.cache/huggingface \
  -p 8000:8000 \
  local/vllm-gpt-oss:20b \
  --api-key ${VLLM_API_KEY}
```

- `--gpus all` + `--ipc=host` recommended.
- First run downloads `openai/gpt-oss-20b` into `/opt/vllm/hf-cache`.

Docker Compose alternative (Compose v2.6+ GPU support):

```yaml
# Save as docker-compose.vllm.yml
services:
  vllm-gpt-oss-20b:
    image: local/vllm-gpt-oss:20b
    build:
      context: .
      dockerfile: Dockerfile.vllm-gpt-oss
    environment:
      HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN}
      VLLM_API_KEY: ${VLLM_API_KEY}
    volumes:
      - /opt/vllm/hf-cache:/root/.cache/huggingface
    ports:
      - "7777:8000"   # maps container 8000 to host 7777
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["--api-key", "${VLLM_API_KEY}"]
```

Run with: `HF_TOKEN=... VLLM_API_KEY=... docker compose -f docker-compose.vllm.yml up --build -d`. Update `VLLM_BASE_URL` to match your host port (e.g., `http://localhost:7777/v1`).

Logs:

```bash
docker logs -f vllm-gpt-oss-20b
```

### 12.8 Sanity checks

List models:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer ${VLLM_API_KEY}"
```

Chat test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      { "role": "user", "content": "Say hello from gpt-oss-20b running on vLLM." }
    ],
    "max_tokens": 64
  }'
```

Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://mlbox.local:8000/v1",
    api_key="local-secret-key",
)

resp = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Test message from my agent backend."},
    ],
    max_tokens=128,
)

print(resp.choices[0].message.content)
```

### 12.9 BGE-M3 embeddings (backend)

- Install: `pip install -U FlagEmbedding`
- Minimal helper:

```python
from FlagEmbedding import BGEM3FlagModel

bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def embed_dense(texts: list[str]) -> list[list[float]]:
    output = bge_model.encode(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return output["dense_vecs"]
```

- For hybrid or ColBERT-style retrieval, set `return_sparse=True` or `return_colbert_vecs=True`.

### 12.10 Backend wiring

Environment:

```
VLLM_BASE_URL=http://mlbox.local:8000/v1
VLLM_API_KEY=local-secret-key
VLLM_MODEL=openai/gpt-oss-20b
EMBED_MODEL_ID=BAAI/bge-m3
```

Client example:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ["VLLM_BASE_URL"],
    api_key=os.environ["VLLM_API_KEY"],
)

def llm_chat(messages):
    return client.chat.completions.create(
        model=os.environ["VLLM_MODEL"],
        messages=messages,
        max_tokens=512,
    )
```

RAG flow:
- User message -> `embed_dense` -> search vector/graph store.
- Build context -> call `llm_chat` with history + context.
- Return reply to frontend.

### 12.11 Troubleshooting

- OOM or crash: gpt-oss-20b targets ~16GB VRAM; reduce max sequence length or use larger GPU.
- No CUDA device in container: re-run `docker run --gpus all ... nvidia-smi`; ensure `nvidia-ctk runtime configure --runtime=docker` and restart Docker.
- HF auth errors: set `HUGGING_FACE_HUB_TOKEN` with access to both `openai/gpt-oss-20b` and `BAAI/bge-m3`; accept licenses in your HF account.
- Slow first request: first call downloads/warms the model; later requests speed up via vLLM caching.
- Retrieval quality: confirm you are using BGE-M3 (dim 1024) and appropriate context length.

[1]: https://cookbook.openai.com/articles/gpt-oss/run-vllm "How to run gpt-oss with vLLM"
[2]: https://huggingface.co/BAAI/bge-m3 "BAAI/bge-m3"
[3]: https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b"
[4]: https://openai.com/index/introducing-gpt-oss/ "Introducing gpt-oss"
[5]: https://bge-model.com/tutorial/1_Embedding/1.2.1.html "BGE Series — BGE documentation"
[6]: https://zilliz.com/ai-models/bge-m3 "The guide to bge-m3 | BAAI"
[7]: https://docs.vllm.ai/en/latest/deployment/docker/ "Using Docker - vLLM"
