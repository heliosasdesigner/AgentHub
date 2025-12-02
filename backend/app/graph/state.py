from __future__ import annotations

from typing import List

from ..agents.base import Message


class MessagesState:
    """Lightweight holder for LangGraph state."""

    def __init__(self, messages: List[Message] | None = None):
        self.messages = messages or []

    def add(self, message: Message) -> None:
        self.messages.append(message)
