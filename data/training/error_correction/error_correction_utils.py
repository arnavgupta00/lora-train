#!/usr/bin/env python3
"""
Error-correction dataset utilities.

Shared helpers for validating the SQL repair training dataset used by
`training/train_lora.py`.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple


ERROR_CORRECTION_SYSTEM_PROMPT = (
    "You are an expert SQL repair assistant. Given schema, question, hints, "
    "broken SQL, and optional database error, output the corrected SQL query only."
)


def get_error_correction_system_prompt_hash() -> str:
    """Get a short hash of the canonical repair system prompt."""
    return hashlib.sha256(ERROR_CORRECTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


class ErrorCorrectionValidationError(Exception):
    """Raised when an example does not follow the repair prompt contract."""


def validate_error_correction_prompt(
    system: str,
    user: str,
    assistant: str = "",
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate one repair prompt/response pair."""
    errors: List[str] = []

    if system.strip() != ERROR_CORRECTION_SYSTEM_PROMPT:
        errors.append("System prompt does not match error-correction contract")

    required_sections = ["Schema:", "Hints:", "Question:", "Broken SQL:"]
    for section in required_sections:
        if section not in user:
            errors.append(f"User prompt missing '{section}' section")

    section_positions = {section: user.find(section) for section in required_sections}
    ordered_sections = required_sections
    for prev, curr in zip(ordered_sections, ordered_sections[1:]):
        prev_idx = section_positions.get(prev, -1)
        curr_idx = section_positions.get(curr, -1)
        if prev_idx != -1 and curr_idx != -1 and prev_idx > curr_idx:
            errors.append(f"'{prev}' must come before '{curr}'")

    if "Broken SQL:" in user:
        broken_sql = user.split("Broken SQL:", 1)[1].strip()
        if broken_sql.startswith("Error:") or broken_sql.startswith("Failure Type:"):
            errors.append("Broken SQL section is empty")

    if "Schema:" in user and "Hints:" in user:
        schema_section = user.split("Schema:", 1)[1].split("Hints:", 1)[0].strip()
        if not schema_section:
            errors.append("Schema section is empty")

    if not assistant.strip():
        errors.append("Assistant response is empty")

    forbidden_output_markers = ("```", "<think>", "</think>")
    for marker in forbidden_output_markers:
        if marker in assistant:
            errors.append(f"Assistant response contains forbidden marker '{marker}'")

    prose_markers = (
        "Here is",
        "Here's",
        "The corrected SQL",
        "Explanation:",
        "Corrected SQL:",
    )
    if any(marker in assistant for marker in prose_markers):
        errors.append("Assistant response contains explanatory prose instead of SQL only")

    is_valid = not errors
    if strict and not is_valid:
        raise ErrorCorrectionValidationError(
            "Error-correction validation failed: " + "; ".join(errors)
        )

    return is_valid, errors


def validate_error_correction_messages(
    messages: List[Dict[str, str]],
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate chat-format messages for the repair dataset."""
    errors: List[str] = []

    if len(messages) < 3:
        errors.append("Messages must include system, user, and assistant roles")
        if strict:
            raise ErrorCorrectionValidationError(
                "Error-correction validation failed: " + "; ".join(errors)
            )
        return False, errors

    expected_roles = ["system", "user", "assistant"]
    for idx, role in enumerate(expected_roles):
        if messages[idx].get("role") != role:
            errors.append(f"Message {idx + 1} must have role '{role}'")

    if errors:
        if strict:
            raise ErrorCorrectionValidationError(
                "Error-correction validation failed: " + "; ".join(errors)
            )
        return False, errors

    return validate_error_correction_prompt(
        messages[0].get("content", ""),
        messages[1].get("content", ""),
        messages[2].get("content", ""),
        strict=strict,
    )
