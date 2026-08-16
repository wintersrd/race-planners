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
            aid_station_tiers={5.3: "water_only", 9.1: "water_only"},
            baseline_peak_temp_c=16.0,
            event_month=9,  # September
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
            aid_stops_km=[16.9, 39.9, 47.4],
            aid_station_tiers={
                16.9: "water_only",
                39.9: "water_only",
                47.4: "water_only",
            },
            start_time_local="12:30",
            baseline_peak_temp_c=18.0,
            event_month=9,  # September
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
            aid_stops_km=[20.0, 37.1, 50.3, 70.7, 75.3, 83.1],
            aid_station_tiers={
                20.0: "water_only",
                37.1: "full_service",
                50.3: "water_only",
                70.7: "full_service",
                75.3: "water_only",
                83.1: "water_only",
            },
            start_time_local="06:30",
            baseline_peak_temp_c=20.0,
            event_month=9,  # September
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
            aid_stops_km=[25.3, 38.5, 53.3, 75.0, 94.8, 111.9, 125.1, 145.4, 150.5, 158.0],
            aid_station_tiers={
                25.3: "water_only",
                38.5: "full_service",
                53.3: "water_only",
                75.0: "full_service",
                94.8: "water_only",
                111.9: "full_service",
                125.1: "water_only",
                145.4: "full_service",
                150.5: "water_only",
                158.0: "water_only",
            },
            start_time_local="17:00",
            baseline_peak_temp_c=20.0,
            event_month=9,  # September
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
            start_time_local="09:00",
            baseline_peak_temp_c=16.0,
            event_month=5,  # May
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
            aid_stops_km=[18.0, 38.0, 45.0],
            aid_station_tiers={
                18.0: "standard",
                38.0: "standard",
                45.0: "water_only",
            },
            start_time_local="11:00",
            baseline_peak_temp_c=20.0,
            event_month=6,  # June
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
