from __future__ import annotations

from typing import Awaitable, Callable

from ..base import AgentInput, AgentOutput, Message


def build_rag_faiss_agent() -> Callable[[AgentInput], Awaitable[AgentOutput]]:
    async def run_agent(agent_input: AgentInput) -> AgentOutput:
        question = agent_input.messages[-1].content if agent_input.messages else "Hello!"
        answer = (
            "Placeholder response from rag-faiss agent. "
            "Connect a real Faiss retriever to replace this stub."
        )
        return AgentOutput(
            reply=Message(role="assistant", content=f"{answer}\n\nEcho: {question}"),
            metadata={"num_docs": 0},
        )

    return run_agent
