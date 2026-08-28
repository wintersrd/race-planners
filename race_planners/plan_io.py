from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from race_planners.event_catalog import CURATED_COURSES_DIR, get_curated_event_by_course_id
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


def ensure_gpx_exists_for_plan(plan_payload: dict[str, Any]) -> Path:
    gpx_filename = str(plan_payload.get("gpx_filename", ""))
    event = get_curated_event_by_course_id(str(plan_payload.get("course_id", "")))
    if event is not None and event.gpx_relative_path.name == gpx_filename:
        candidate = CURATED_COURSES_DIR / event.gpx_relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing course file: {gpx_filename}. This plan requires a bundled curated course."
    )


def load_plan_into_state(
    plan_json: str,
    current_state: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    try:
        payload = import_plan_json(plan_json)
        ensure_gpx_exists_for_plan(payload)
    except (ValueError, FileNotFoundError) as exc:
        return current_state, str(exc)

    updated_state = dict(current_state)
    updated_state["general_course_id"] = str(payload["course_id"])
    updated_state["general_config"] = dict(payload["config"])
    if "athlete_profile" in payload:
        updated_state["general_athlete_profile"] = dict(payload["athlete_profile"])
    matched_event = get_curated_event_by_course_id(str(payload["course_id"]))
    if matched_event is not None:
        updated_state["general_event_id"] = matched_event.event_id
    return updated_state, None
