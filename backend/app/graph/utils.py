from __future__ import annotations

from typing import Iterable, List

from ..agents.base import Message


def last_user_message(messages: Iterable[Message]) -> Message | None:
    for msg in reversed(list(messages)):
        if msg.role == "user":
            return msg
    return None
