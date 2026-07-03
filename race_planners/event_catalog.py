from __future__ import annotations

from pathlib import Path

from race_planners.models import CuratedEvent, EventTemplate


_EVENT_TEMPLATES: tuple[EventTemplate, ...] = (
    EventTemplate(
        template_id="road_half",
        label="Road Half Marathon",
        race_model="half_marathon",
        terrain="road",
    ),
    EventTemplate(
        template_id="road_marathon",
        label="Road Marathon",
        race_model="road_marathon",
        terrain="road",
    ),
    EventTemplate(
        template_id="trail_short",
        label="Short Trail",
        race_model="technical_trail_ultra",
        terrain="trail",
    ),
    EventTemplate(
        template_id="trail_ultra",
        label="Trail Ultra",
        race_model="technical_trail_ultra",
        terrain="trail",
    ),
    EventTemplate(
        template_id="trail_very_long_ultra",
        label="Very Long Trail Ultra",
        race_model="technical_trail_ultra",
        terrain="trail",
    ),
)


def list_event_templates() -> list[EventTemplate]:
    return list(_EVENT_TEMPLATES)


def get_event_template(template_id: str) -> EventTemplate:
    for template in _EVENT_TEMPLATES:
        if template.template_id == template_id:
            return template
    raise ValueError(f"Unknown event template: {template_id}")


def _curated_event_definitions() -> tuple[CuratedEvent, ...]:
    return (
        CuratedEvent(
            event_id="semi-marathon-finistere",
            name="Semi-Marathon du Finistere",
            short_name="Finistere Half",
            template_id="road_half",
            course_id="semi-marathon-finistere",
            gpx_relative_path=Path("semi-marathon-finistere/semi-marathon-du-finistere.gpx"),
            race_model="half_marathon",
            terrain="road",
            aid_stops_km=[5.3, 9.1, 14.5],
        ),
        CuratedEvent(
            event_id="grf56",
            name="Grand Raid du Finistere 56",
            short_name="GRF56",
            template_id="trail_ultra",
            course_id="grf56",
            gpx_relative_path=Path("semi-marathon-finistere/2026-grf56.gpx"),
            race_model="technical_trail_ultra",
            terrain="trail",
        ),
        CuratedEvent(
            event_id="grf92",
            name="Grand Raid du Finistere 92",
            short_name="GRF92",
            template_id="trail_ultra",
            course_id="grf92",
            gpx_relative_path=Path("semi-marathon-finistere/2026-grf92.gpx"),
            race_model="technical_trail_ultra",
            terrain="trail",
        ),
        CuratedEvent(
            event_id="grf166",
            name="Grand Raid du Finistere 166",
            short_name="GRF166",
            template_id="trail_ultra",
            course_id="grf166",
            gpx_relative_path=Path("semi-marathon-finistere/2026-grf166.gpx"),
            race_model="technical_trail_ultra",
            terrain="trail",
        ),
        CuratedEvent(
            event_id="marathon-etoiles-baie",
            name="Marathon des Etoiles de la Baie",
            short_name="Etoiles Marathon",
            template_id="road_marathon",
            course_id="marathon-etoiles-baie",
            gpx_relative_path=Path("semi-marathon-finistere/marathon-des-etoiles-de-la-baie.gpx"),
            race_model="road_marathon",
            terrain="road",
        ),
        CuratedEvent(
            event_id="trail-odet-ultra",
            name="Trail de l'Odet Ultra",
            short_name="Odet Ultra",
            template_id="trail_ultra",
            course_id="trail-odet-ultra",
            gpx_relative_path=Path("semi-marathon-finistere/trail-de-l-odet-ultra.gpx"),
            race_model="technical_trail_ultra",
            terrain="trail",
        ),
    )


def list_curated_events(repo_root: Path) -> list[CuratedEvent]:
    curated_events: list[CuratedEvent] = []
    for event in _curated_event_definitions():
        if (repo_root / event.gpx_relative_path).exists():
            curated_events.append(event)
    return curated_events


def get_curated_event(repo_root: Path, event_id: str) -> CuratedEvent:
    for event in list_curated_events(repo_root):
        if event.event_id == event_id:
            return event
    raise ValueError(f"Unknown event id: {event_id}")


def get_curated_event_by_course_id(repo_root: Path, course_id: str) -> CuratedEvent | None:
    for event in list_curated_events(repo_root):
        if event.course_id == course_id:
            return event
    return None
