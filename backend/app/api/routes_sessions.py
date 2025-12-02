from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..persistence.db import get_session
from ..persistence.models import ConversationRecord
from ..services.history_service import fetch_messages

router = APIRouter()


@router.get("/{session_id}", response_model=dict)
def get_session_history(session_id: str, session: Session = Depends(get_session)) -> dict:
    conversation = session.get(ConversationRecord, session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = fetch_messages(session, session_id)
    return {"session_id": session_id, "agent_id": conversation.agent_id, "messages": messages}
