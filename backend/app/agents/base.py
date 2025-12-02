from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    role: Role
    content: str


class AgentInput(BaseModel):
    agent_id: str
    session_id: str
    messages: List[Message]
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    reply: Message
    metadata: Dict[str, Any] = Field(default_factory=dict)
