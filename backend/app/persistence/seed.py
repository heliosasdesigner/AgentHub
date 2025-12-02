"""
Database seeding utilities
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy.orm import Session

from .models import SystemPromptRecord


def seed_template_prompts(session: Session) -> None:
    """
    Seed the database with template system prompts if they don't already exist.

    Loads prompts from assets/template_prompts.json and inserts them only if
    they don't already exist in the database (checked by ID).
    """
    # Get the path to the template prompts JSON file
    current_dir = Path(__file__).parent
    template_file = current_dir / "assets" / "template_prompts.json"

    if not template_file.exists():
        print(f"Warning: Template prompts file not found at {template_file}")
        return

    # Load template prompts from JSON
    try:
        with open(template_file, "r", encoding="utf-8") as f:
            templates = json.load(f)
    except Exception as e:
        print(f"Error loading template prompts: {e}")
        return

    # Insert each template if it doesn't exist
    inserted_count = 0
    skipped_count = 0

    for template in templates:
        # Check if prompt with this ID already exists
        existing = session.query(SystemPromptRecord).filter(
            SystemPromptRecord.id == template["id"]
        ).first()

        if existing:
            skipped_count += 1
            continue

        # Insert new template prompt
        prompt = SystemPromptRecord(
            id=template["id"],
            name=template["name"],
            content=template["content"],
            user_id=template.get("user_id")
        )
        session.add(prompt)
        inserted_count += 1

    # Commit all inserts
    if inserted_count > 0:
        session.commit()
        print(f"Seeded {inserted_count} template system prompts")

    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing template prompts")
