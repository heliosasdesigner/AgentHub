from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Union

from pydantic import BaseModel

from .base import AgentInput, AgentOutput
from .cognee_agent.builder import build_cognee_agent
from .graphrag_neo4j.builder import build_graphrag_neo4j_agent
from .leann_agent.builder import build_leann_agent
from .rag_faiss.builder import build_rag_faiss_agent

AgentRunner = Union[
    Callable[[AgentInput], AgentOutput],
    Callable[[AgentInput], Awaitable[AgentOutput]],
]


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str


AGENT_REGISTRY: Dict[str, AgentRunner] = {}

AGENT_METADATA: List[AgentInfo] = [
    AgentInfo(
        id="rag-faiss",
        name="RAG (Faiss)",
        description="Vector RAG baseline backed by a Faiss index.",
    ),
    AgentInfo(
        id="graphrag-neo4j",
        name="GraphRAG (Neo4j)",
        description="Graph-based retrieval using Neo4j property graph.",
    ),
    AgentInfo(
        id="cognee",
        name="Cognee GraphRAG",
        description="GraphRAG via Cognee pipeline and storage.",
    ),
    AgentInfo(
        id="leann",
        name="LEANN",
        description="Lightweight vector database with graph-based selective recomputation.",
    ),
]


def init_agents() -> None:
    """Build and register available agent runners."""
    if AGENT_REGISTRY:
        return

    AGENT_REGISTRY["rag-faiss"] = build_rag_faiss_agent()
    AGENT_REGISTRY["graphrag-neo4j"] = build_graphrag_neo4j_agent()
    AGENT_REGISTRY["cognee"] = build_cognee_agent()
    AGENT_REGISTRY["leann"] = build_leann_agent()


def list_agents() -> List[AgentInfo]:
    return AGENT_METADATA


def get_agent_runner(agent_id: str) -> AgentRunner:
    runner = AGENT_REGISTRY.get(agent_id)
    if not runner:
        raise KeyError(f"Unknown agent id '{agent_id}'.")
    return runner
