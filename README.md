# AgentHub

An AI agent experiment platform that enables you to compare and interact with multiple AI agent architectures through a unified interface. Built with FastAPI, React, and LangChain.

> **⚠️ Work in Progress**: This project is currently under active development. The backend is still being developed and may have incomplete features or breaking changes. The frontend is functional but may not fully integrate with all backend features yet.

## 🚀 Features

- **Multi-Agent Support**: Compare responses from different AI agent architectures simultaneously
- **Modern Web UI**: React-based frontend with real-time chat interface
- **Pluggable Architecture**: Easy to add new agent implementations
- **Multiple Storage Backends**: Support for vector databases (Faiss), graph databases (Neo4j), and more
- **Flexible LLM Routing**: Support for OpenAI, LM Studio, and vLLM endpoints
- **Session Management**: Track conversations and agent performance
- **System Prompts**: Manage and apply system prompts across agents
- **Visual Workflows**: Mermaid diagrams for agent processing flows

## 📋 Available Agents

- **rag-faiss**: Vector RAG baseline backed by a Faiss index
- **graphrag-neo4j**: Graph-based retrieval using Neo4j property graph
- **cognee**: GraphRAG via Cognee pipeline and storage
- **leann**: Lightweight vector database with graph-based selective recomputation

## 🏗️ Architecture

```
┌─────────────┐
│   React     │  Frontend (Vite + TypeScript)
│   Frontend  │
└──────┬──────┘
       │ HTTP
┌──────▼──────┐
│   FastAPI   │  Backend API
│   Backend   │
└──────┬──────┘
       │
   ┌───┴───┬──────────┬──────────┐
   │       │          │          │
┌──▼──┐ ┌──▼──┐  ┌───▼───┐  ┌───▼──┐
│Faiss│ │Neo4j│  │Cognee │  │SQLite│
└─────┘ └─────┘  └───────┘  └──────┘
```

The platform consists of:
- **Backend**: FastAPI application with agent registry and routing
- **Frontend**: React application with TypeScript and Tailwind CSS
- **Storage**: Multiple backends (SQLite/Postgres for conversations, Neo4j for graphs, Faiss for vectors)

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for database operations
- **LangChain** - LLM framework and agent building
- **Pydantic** - Data validation and settings management
- **Neo4j** - Graph database driver
- **Uvicorn** - ASGI server

### Frontend
- **React 19** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **Mermaid** - Diagram rendering

## 📦 Prerequisites

- **Python 3.11+**
- **Node.js 18+** (or pnpm)
- **Docker** (optional, for Neo4j)
- **uv** (Python package manager) - Install via: `pip install uv`

## 🔧 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd AgentHub
```

### 2. Backend Setup

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
pnpm install  # or npm install
```

### 4. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Core
APP_ENV=dev
BACKEND_HOST=0.0.0.0
BACKEND_PORT=11111

# Database
DB_URL=sqlite:///./agent_lab.db
# For PostgreSQL: DB_URL=postgresql+psycopg2://user:password@localhost:5432/agent_lab

# LLM Configuration
# Option 1: OpenAI
OPENAI_API_KEY=sk-...

# Option 2: LM Studio (local)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=
LMSTUDIO_MODEL=

# Option 3: vLLM
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-secret-key
VLLM_MODEL=openai/gpt-oss-20b

# Neo4j (for graphrag-neo4j agent)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpassword

# Embeddings
EMBED_MODEL_ID=BAAI/bge-m3

# Cognee
COGNEE_DATA_DIR=./cognee_data
```

## 🚀 Running the Application

> **Note**: The backend is still under development. Some features may not be fully implemented or may change.

### Start Neo4j (Optional, for graphrag-neo4j agent)

```bash
docker compose up -d
```

This starts Neo4j on:
- HTTP: `http://localhost:7474`
- Bolt: `bolt://localhost:7687`

### Start the Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 11111
```

The API will be available at `http://localhost:11111`

API documentation: `http://localhost:11111/docs`

> **Development Status**: Backend APIs and agent implementations are being actively developed. Expect API changes and incomplete features.

### Start the Frontend

```bash
cd frontend
pnpm dev
```

The frontend will be available at `http://localhost:5173`

## 📚 API Endpoints

> **Note**: API endpoints are subject to change as the backend is still under development.

### Agents
- `GET /agents` - List all available agents
- `GET /agents/{agent_id}` - Get agent details

### Chat
- `POST /chat` - Send a message to one or more agents
  ```json
  {
    "agent_id": "rag-faiss",
    "session_id": "uuid",
    "message": "Your message here",
    "system_prompt": "Optional system prompt"
  }
  ```

### Sessions
- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session details
- `DELETE /sessions/{session_id}` - Delete a session

### Prompts
- `GET /prompts` - List all system prompts
- `POST /prompts` - Create a new prompt
- `GET /prompts/{prompt_id}` - Get prompt details
- `PUT /prompts/{prompt_id}` - Update a prompt
- `DELETE /prompts/{prompt_id}` - Delete a prompt

## 🧪 Testing

Run the test suite:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run tests
pytest
```

Tests use an isolated SQLite database and mock external dependencies.

## 🏗️ Development

### Project Structure

```
AgentHub/
├── backend/
│   └── app/
│       ├── api/              # HTTP route handlers
│       ├── agents/            # Agent implementations
│       │   ├── rag_faiss/
│       │   ├── graphrag_neo4j/
│       │   ├── cognee_agent/
│       │   └── leann_agent/
│       ├── graph/             # LangGraph utilities
│       ├── persistence/       # Database models and operations
│       └── services/          # Business logic
├── frontend/
│   └── src/
│       ├── components/        # React components
│       ├── pages/             # Page components
│       └── lib/               # Utilities
├── tests/                     # Test suite
└── docs/                      # Documentation
```

### Adding a New Agent

1. Create agent directory under `backend/app/agents/your_agent/`
2. Implement `builder.py` with a `build_your_agent()` function returning `AgentRunner`
3. Add agent metadata to `AGENT_METADATA` list in `registry.py`
4. Register builder in `init_agents()` function

See `AGENTS.md` for detailed guidelines.

### Code Style

- Python 3.11+, 4-space indentation
- Type hints throughout
- Descriptive function and module names
- Comments only for non-obvious logic

## 🔒 Security

- Never commit `.env` files or secrets
- All sensitive configuration is loaded from environment variables
- Database credentials should be set via environment variables
- Review `.gitignore` to ensure sensitive files are excluded

## 📖 Documentation

- `AGENTS.md` - Agent development guidelines
- `docs/ai-agent-implementation.md` - Detailed implementation guide
- `CLAUDE.md` - Development notes for AI assistants

## 🤝 Contributing

This project is actively being developed. Contributions are welcome, but please note:

- The backend is still under active development
- API contracts may change
- Some features may be incomplete or experimental

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request with a clear description of changes

## 📝 License

MIT

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://www.langchain.com/)
- [React](https://react.dev/)
- [Neo4j](https://neo4j.com/)
- [Faiss](https://github.com/facebookresearch/faiss)

