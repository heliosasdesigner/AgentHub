from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..agents.registry import AgentInfo, list_agents
from ..persistence.db import get_session
from ..persistence.models import AgentRecord

router = APIRouter()


def _sync_agents_table(session: Session) -> None:
    existing = {row.id for row in session.query(AgentRecord.id).all()}
    created = False
    for agent in list_agents():
        if agent.id in existing:
            continue
        record = AgentRecord(id=agent.id, name=agent.name, description=agent.description)
        session.add(record)
        created = True
    if created:
        session.commit()


@router.get("", response_model=list[AgentInfo])
def get_agents(session: Session = Depends(get_session)) -> list[AgentInfo]:
    _sync_agents_table(session)
    return list_agents()
