from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agents.registry import init_agents
from .persistence.db import init_db, SessionLocal
from .persistence.seed import seed_template_prompts
from .api import routes_agents, routes_chat, routes_sessions, routes_prompts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    init_db()

    # Seed template prompts if they don't exist
    session = SessionLocal()
    try:
        seed_template_prompts(session)
    finally:
        session.close()

    # Initialize agents
    init_agents()

    yield


app = FastAPI(
    title="AI Agent Experiment Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_agents.router, prefix="/agents", tags=["agents"])
app.include_router(routes_chat.router, tags=["chat"])
app.include_router(routes_sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(routes_prompts.router, tags=["prompts"])
