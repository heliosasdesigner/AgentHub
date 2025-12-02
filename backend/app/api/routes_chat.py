from __future__ import annotations

import inspect

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..agents.base import AgentInput, AgentOutput
from ..agents.registry import get_agent_runner, list_agents
from ..persistence.db import get_session
from ..persistence.models import AgentRecord
from ..services.history_service import add_message, ensure_conversation, finish_run, start_run

router = APIRouter()


def _ensure_agent_row(session: Session, agent_id: str) -> None:
    if session.get(AgentRecord, agent_id):
        return

    agent = next((a for a in list_agents() if a.id == agent_id), None)
    if not agent:
        return

    session.add(AgentRecord(id=agent.id, name=agent.name, description=agent.description))
    session.commit()


@router.post("/chat", response_model=AgentOutput)
async def chat(input_payload: AgentInput, session: Session = Depends(get_session)) -> AgentOutput:
    try:
        runner = get_agent_runner(input_payload.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    _ensure_agent_row(session, input_payload.agent_id)
    try:
        ensure_conversation(session, input_payload.session_id, input_payload.agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for message in input_payload.messages:
        add_message(session, input_payload.session_id, message)

    run = start_run(session, input_payload.session_id, input_payload.agent_id)

    try:
        result = runner(input_payload)
        output: AgentOutput = await result if inspect.isawaitable(result) else result
        add_message(session, input_payload.session_id, output.reply)
        finish_run(session, run.id, success=True, metrics=output.metadata)
        return output
    except Exception as exc:  # noqa: BLE001
        finish_run(session, run.id, success=False, metrics={"error": str(exc)})
        raise
