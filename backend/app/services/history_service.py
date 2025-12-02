from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from ..agents.base import Message
from ..persistence.models import ConversationRecord, MessageRecord, RunRecord


def ensure_conversation(session: Session, conversation_id: str, agent_id: str) -> ConversationRecord:
    conversation = session.get(ConversationRecord, conversation_id)
    if conversation:
        if conversation.agent_id != agent_id:
            raise ValueError("Conversation exists with a different agent_id.")
        return conversation

    conversation = ConversationRecord(id=conversation_id, agent_id=agent_id)
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


def add_message(session: Session, conversation_id: str, message: Message) -> MessageRecord:
    record = MessageRecord(
        conversation_id=conversation_id,
        role=message.role,
        content=message.content,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def fetch_messages(session: Session, conversation_id: str) -> List[Message]:
    rows = (
        session.query(MessageRecord)
        .filter(MessageRecord.conversation_id == conversation_id)
        .order_by(MessageRecord.created_at.asc())
        .all()
    )
    return [Message(role=row.role, content=row.content) for row in rows]


def start_run(session: Session, conversation_id: str, agent_id: str) -> RunRecord:
    run = RunRecord(conversation_id=conversation_id, agent_id=agent_id, started_at=datetime.utcnow())
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def finish_run(
    session: Session,
    run_id: int,
    success: bool,
    metrics: Optional[dict] = None,
) -> Optional[RunRecord]:
    run = session.get(RunRecord, run_id)
    if not run:
        return None

    run.finished_at = datetime.utcnow()
    run.success = success
    run.metrics_json = json.dumps(metrics or {})
    session.commit()
    session.refresh(run)
    return run
