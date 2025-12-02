from __future__ import annotations

from typing import Awaitable, Callable

from ..base import AgentInput, AgentOutput, Message


def build_leann_agent() -> Callable[[AgentInput], Awaitable[AgentOutput]]:
    async def run_agent(agent_input: AgentInput) -> AgentOutput:
        user_message = agent_input.messages[-1].content if agent_input.messages else ""
        reply_text = (
            "LEANN agent stub. LEANN is a lightweight vector database with graph-based selective recomputation. "
            "Connect LEANN implementation to replace this stub."
            f"\n\nQuestion: {user_message}"
        )
        return AgentOutput(
            reply=Message(role="assistant", content=reply_text),
            metadata={"leann_used": False},
        )

    return run_agent
