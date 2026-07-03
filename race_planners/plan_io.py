from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from race_planners.models import PacingConfig

PLAN_SCHEMA_VERSION = 1


def export_plan_json(
    course_id: str,
    gpx_filename: str,
    config: PacingConfig,
    athlete_profile: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "course_id": course_id,
        "gpx_filename": gpx_filename,
        "config": asdict(config),
    }
    if athlete_profile is not None:
        payload["athlete_profile"] = athlete_profile
    return json.dumps(payload, indent=2, sort_keys=True)


def import_plan_json(plan_json: str) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(plan_json))
    schema_version = payload.get("schema_version")
    if schema_version != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported plan schema version: {schema_version}. Expected {PLAN_SCHEMA_VERSION}."
        )
    return payload


def ensure_gpx_exists_for_plan(plan_payload: dict[str, Any], gpx_search_roots: list[Path]) -> Path:
    gpx_filename = str(plan_payload.get("gpx_filename", ""))
    for root in gpx_search_roots:
        candidate = root / gpx_filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing course file: {gpx_filename}. Restore this course file in the repository to reload the saved plan."
    )
