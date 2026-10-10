import json
import logging
from typing import Any, Dict, Union

from pydantic import ValidationError

from app.database import clear_and_ingest_knowledge
from app.models import KnowledgePayload

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Base exception for knowledge ingestion operations."""
    pass


class IngestionValidationError(IngestionError):
    """Raised when JSON parsing, required fields, or validation rules fail."""
    pass


def parse_and_validate_knowledge(
    raw_content: Union[str, bytes, Dict[str, Any]]
) -> KnowledgePayload:
    """
    Parses and strictly validates knowledge JSON content.
    Ensures correct types, non-empty values, positive employee counts,
    non-negative leave days, and uniqueness of department/leave policy names.
    """
    # 1. Parse JSON if string or bytes
    if isinstance(raw_content, (str, bytes)):
        if isinstance(raw_content, bytes):
            try:
                decoded_str = raw_content.decode("utf-8")
            except UnicodeDecodeError as e:
                raise IngestionValidationError(f"File encoding error: {str(e)}") from e
        else:
            decoded_str = raw_content

        if not decoded_str or not decoded_str.strip():
            raise IngestionValidationError("Uploaded knowledge file is empty.")

        try:
            data = json.loads(decoded_str)
        except json.JSONDecodeError as e:
            raise IngestionValidationError(f"Malformed JSON: {e.msg} at line {e.lineno} column {e.colno}") from e
    elif isinstance(raw_content, dict):
        data = raw_content
    else:
        raise IngestionValidationError("Invalid payload format: expected JSON string, bytes, or dictionary.")

    if not isinstance(data, dict):
        raise IngestionValidationError("Knowledge JSON root must be an object/dictionary.")

    # 2. Validate using Pydantic KnowledgePayload model
    try:
        validated_payload = KnowledgePayload.model_validate(data)
        return validated_payload
    except ValidationError as e:
        # Extract the first informative validation error message
        errors = e.errors()
        err_messages = []
        for err in errors:
            loc = " -> ".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            err_messages.append(f"{loc}: {msg}" if loc else msg)
        first_err = err_messages[0] if err_messages else "Validation failed"
        raise IngestionValidationError(f"Knowledge validation error: {first_err}") from e


def ingest_knowledge_from_content(
    raw_content: Union[str, bytes, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validates and ingests knowledge data into SQLite.
    Atomically clears previous records and writes newly validated records using
    parameterized SQL queries only.
    Returns an import summary.
    """
    validated = parse_and_validate_knowledge(raw_content)

    company_dict = validated.company.model_dump()
    departments_list = [d.model_dump() for d in validated.departments]
    leave_policy_list = [l.model_dump() for l in validated.leave_policy]
    onboarding_items_list = (
        [item.model_dump() for item in validated.onboarding_items]
        if validated.onboarding_items
        else None
    )

    # Atomically replace knowledge in SQLite
    clear_and_ingest_knowledge(
        company=company_dict,
        departments=departments_list,
        leave_policy=leave_policy_list,
        onboarding_items=onboarding_items_list,
    )

    logger.info(
        "Successfully ingested knowledge for '%s' (%d departments, %d leave policies, %d onboarding items).",
        company_dict["name"],
        len(departments_list),
        len(leave_policy_list),
        len(onboarding_items_list) if onboarding_items_list else 0,
    )

    return {
        "status": "success",
        "message": "Knowledge imported successfully",
        "company": company_dict["name"],
        "departments_imported": len(departments_list),
        "leave_policies_imported": len(leave_policy_list),
        "onboarding_items_imported": len(onboarding_items_list) if onboarding_items_list else 0,
    }
