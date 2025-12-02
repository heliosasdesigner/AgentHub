"""
API routes for system prompt management
"""
from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.persistence.db import get_session
from app.persistence.models import SystemPromptRecord

router = APIRouter()


class SystemPromptCreate(BaseModel):
    name: str
    content: str
    user_id: str | None = None


class SystemPromptUpdate(BaseModel):
    name: str | None = None
    content: str | None = None


class SystemPromptResponse(BaseModel):
    id: str
    name: str
    content: str
    user_id: str | None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("/prompts", response_model=List[SystemPromptResponse])
def list_prompts(session: Session = Depends(get_session)):
    """List all system prompts"""
    prompts = session.query(SystemPromptRecord).order_by(
        SystemPromptRecord.created_at.desc()
    ).all()
    return [
        SystemPromptResponse(
            id=p.id,
            name=p.name,
            content=p.content,
            user_id=p.user_id,
            created_at=p.created_at.isoformat(),
            updated_at=p.updated_at.isoformat(),
        )
        for p in prompts
    ]


@router.get("/prompts/{prompt_id}", response_model=SystemPromptResponse)
def get_prompt(prompt_id: str, session: Session = Depends(get_session)):
    """Get a specific system prompt by ID"""
    prompt = session.query(SystemPromptRecord).filter(
        SystemPromptRecord.id == prompt_id
    ).first()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return SystemPromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        user_id=prompt.user_id,
        created_at=prompt.created_at.isoformat(),
        updated_at=prompt.updated_at.isoformat(),
    )


@router.post("/prompts", response_model=SystemPromptResponse)
def create_prompt(prompt: SystemPromptCreate, session: Session = Depends(get_session)):
    """Create a new system prompt"""
    new_prompt = SystemPromptRecord(
        id=str(uuid.uuid4()),
        name=prompt.name,
        content=prompt.content,
        user_id=prompt.user_id,
    )
    session.add(new_prompt)
    session.commit()
    session.refresh(new_prompt)

    return SystemPromptResponse(
        id=new_prompt.id,
        name=new_prompt.name,
        content=new_prompt.content,
        user_id=new_prompt.user_id,
        created_at=new_prompt.created_at.isoformat(),
        updated_at=new_prompt.updated_at.isoformat(),
    )


@router.put("/prompts/{prompt_id}", response_model=SystemPromptResponse)
def update_prompt(prompt_id: str, prompt: SystemPromptUpdate, session: Session = Depends(get_session)):
    """Update an existing system prompt"""
    existing_prompt = session.query(SystemPromptRecord).filter(
        SystemPromptRecord.id == prompt_id
    ).first()

    if not existing_prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    if prompt.name is not None:
        existing_prompt.name = prompt.name
    if prompt.content is not None:
        existing_prompt.content = prompt.content

    session.commit()
    session.refresh(existing_prompt)

    return SystemPromptResponse(
        id=existing_prompt.id,
        name=existing_prompt.name,
        content=existing_prompt.content,
        user_id=existing_prompt.user_id,
        created_at=existing_prompt.created_at.isoformat(),
        updated_at=existing_prompt.updated_at.isoformat(),
    )


@router.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: str, session: Session = Depends(get_session)):
    """Delete a system prompt"""
    prompt = session.query(SystemPromptRecord).filter(
        SystemPromptRecord.id == prompt_id
    ).first()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    session.delete(prompt)
    session.commit()

    return {"message": "Prompt deleted successfully"}
