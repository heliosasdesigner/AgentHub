from __future__ import annotations

from typing import Awaitable, Callable

from ..base import AgentInput, AgentOutput, Message


def build_graphrag_neo4j_agent() -> Callable[[AgentInput], Awaitable[AgentOutput]]:
    async def run_agent(agent_input: AgentInput) -> AgentOutput:
        prompt = agent_input.messages[-1].content if agent_input.messages else ""
        reply_text = (
            "GraphRAG (Neo4j) stub. Wire Neo4j queries and LangGraph here.\n\n"
            f"Question: {prompt}"
        )
        return AgentOutput(
            reply=Message(role="assistant", content=reply_text),
            metadata={"graph_used": False},
        )

    return run_agent
