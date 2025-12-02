from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

# Point DB_URL to an isolated test database before importing app modules.
TEST_DB = Path(__file__).parent / "agent_lab_test.db"
os.environ["DB_URL"] = f"sqlite:///{TEST_DB}"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.agents.base import Message
from backend.app.graph.utils import last_user_message
from backend.app.persistence import db
from backend.app.services.history_service import (
    add_message,
    ensure_conversation,
    fetch_messages,
)


def setup_module(module):
    if TEST_DB.exists():
        TEST_DB.unlink()
    db.init_db()


def teardown_module(module):
    if TEST_DB.exists():
        TEST_DB.unlink()


@contextmanager
def session_scope():
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_ensure_conversation_agent_mismatch():
    with session_scope() as session:
        ensure_conversation(session, "session-1", "rag-faiss")
        with pytest.raises(ValueError):
            ensure_conversation(session, "session-1", "cognee")


def test_add_and_fetch_messages_in_order():
    convo_id = "session-2"
    with session_scope() as session:
        ensure_conversation(session, convo_id, "rag-faiss")
        add_message(session, convo_id, Message(role="user", content="hello"))
        add_message(session, convo_id, Message(role="assistant", content="hi there"))

    with session_scope() as session:
        messages = fetch_messages(session, convo_id)

    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "hi there"


def test_last_user_message_returns_latest_user():
    messages = [
        Message(role="assistant", content="a"),
        Message(role="user", content="u1"),
        Message(role="assistant", content="b"),
        Message(role="user", content="u2"),
    ]
    result = last_user_message(messages)
    assert result is not None
    assert result.content == "u2"
