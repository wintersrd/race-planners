from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackPoint:
    lat: float
    lon: float
    elevation: float
    time: str
    distance_from_start: float = 0.0
    grade_percent: float = 0.0


@dataclass(frozen=True)
class AidStation:
    distance_km: float
    label: str = ""
    source: str = "config"
    waypoint_type: str = ""
    tier: str = "standard"


@dataclass
class Course:
    course_id: str
    name: str
    gpx_path: Path
    aid_stops_km: list[float] = field(default_factory=list)
    aid_stations: list[AidStation] = field(default_factory=list)
    terrain: str = "road"
    event_id: str | None = None
    template_id: str | None = None

    def __post_init__(self) -> None:
        if self.aid_stations and not self.aid_stops_km:
            self.aid_stops_km = [aid_station.distance_km for aid_station in self.aid_stations]
            return

        if self.aid_stops_km and not self.aid_stations:
            self.aid_stations = [
                AidStation(distance_km=distance_km)
                for distance_km in self.aid_stops_km
                if distance_km > 0
            ]


@dataclass
class LoadedCourse:
    course: Course
    trackpoints: list[TrackPoint]
    total_distance_km: float


@dataclass(frozen=True)
class EventTemplate:
    template_id: str
    label: str
    race_model: str
    terrain: str
    supports_finish_time: bool = True
    supports_effort_anchor: bool = True


@dataclass(frozen=True)
class CuratedEvent:
    event_id: str
    name: str
    short_name: str
    template_id: str
    course_id: str
    gpx_relative_path: Path
    race_model: str
    terrain: str
    aid_stops_km: list[float] = field(default_factory=list)
    aid_station_tiers: dict[float, str] = field(default_factory=dict)
    default_input_mode: str = "finish_time"
    start_time_local: str | None = None
    baseline_peak_temp_c: float | None = None
    event_month: int | None = None


@dataclass
class AthleteProfile:
    lt1_hr: int | None = None
    lt1_pace_min_km: float | None = None
    lt2_hr: int | None = None
    lt2_pace_min_km: float | None = None
    best_likely_half_time_min: float | None = None
    best_likely_marathon_time_min: float | None = None
    predictor_half_time_min: float | None = None
    predictor_marathon_time_min: float | None = None
    predictor_source: str | None = None
    flat_trail_slowdown_sec_km: float | None = None
    technical_trail_slowdown_sec_km: float | None = None
    durability_factor: float | None = None
    heat_tolerance: float | None = None
    hill_tolerance: float | None = None
    default_road_split_bias: float | None = None
    default_trail_fade_preset: str | None = None
    default_trail_effort_policy: str | None = None
    body_mass_kg: float | None = None
    sweat_rate_l_hr: float | None = None
    gut_carb_tolerance_g_hr: float | None = None


@dataclass
class PacingConfig:
    race_model: str
    input_mode: str
    target_finish_time_min: float | None = None
    marathon_pace_min_km: float | None = None
    z1_pace_min_km: float | None = None
    z2_pace_min_km: float | None = None
    flat_pace_min_km: float | None = None
    hike_pace_min_km: float | None = None
    climb_hike_threshold_percent: float = 12.0
    descent_caution: str = "medium"
    rest_duration_sec: int = 30
    rest_duration_water_only_sec: int = 240
    rest_duration_standard_sec: int = 480
    rest_duration_full_service_sec: int = 720
    pacing_bias: float = 0.0
    fade_profile_preset: str | None = None
    fade_early_bias: float | None = None
    fade_mid_bias: float | None = None
    fade_late_bias: float | None = None
    race_intent: str | None = None
    effort_policy: str | None = None
    use_hr_guardrail: bool = False
    athlete_lt1_hr: int | None = None
    athlete_lt2_hr: int | None = None
    athlete_lt1_pace_min_km: float | None = None
    athlete_lt2_pace_min_km: float | None = None
    athlete_flat_trail_slowdown_sec_km: float | None = None
    athlete_technical_trail_slowdown_sec_km: float | None = None
    athlete_durability_factor: float | None = None
    athlete_heat_tolerance: float | None = None
    athlete_hill_tolerance: float | None = None
    rpe_target: float | None = None
    hr_cap: int | None = None
    peak_temperature_c: float | None = None
    event_start_time_local: str | None = None
    event_month: int | None = None


@dataclass
class PaceSplit:
    km: float
    actual_pace_min_km: float
    grade_percent: float
    segment_time_min: float
    cumulative_time_min: float


@dataclass
class SegmentSummary:
    segment_type: str
    start_km: float
    end_km: float
    distance_km: float
    start_time_min: float
    end_time_min: float
    avg_grade_percent: float
    avg_pace_min_km: float
    segment_time_min: float
    elevation_gain_m: float = 0.0
    elevation_loss_m: float = 0.0
    block_label: str = ""
    section_name: str = ""


@dataclass
class AidStationEta:
    distance_km: float
    label: str = ""
    source: str = "config"
    waypoint_type: str = ""
    arrival_moving_time_min: float = 0.0
    arrival_elapsed_time_min: float = 0.0
    departure_elapsed_time_min: float = 0.0
    split_from_prev_min: float = 0.0
    split_distance_km: float = 0.0
    actual_pace_min_km: float = 0.0
    suggested_rest_min: float = 0.0
    elevation_gain_m: float = 0.0
    elevation_loss_m: float = 0.0


@dataclass
class PlanResult:
    splits: list[PaceSplit]
    segments: list[SegmentSummary]
    aid_arrival_times_min: list[float]
    aid_station_etas: list[AidStationEta]
    moving_time_min: float
    total_rest_time_min: float
    total_time_min: float
    total_distance_km: float
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FuelingBlock:
    start_km: float
    end_km: float
    distance_km: float
    duration_min: float
    kcal_burned: float
    carb_target_g: float
    carb_planned_g: float
    fluid_target_l: float
    carry_items: list[str] = field(default_factory=list)
    aid_station_tier: str = "standard"
    on_site_kcal: float = 0.0
    on_site_carb_g: float = 0.0
    cumulative_carb_deficit_g: float = 0.0


@dataclass
class FuelingPlan:
    total_kcal: float
    avg_kcal_hr: float
    total_carb_target_g: float
    total_carb_planned_g: float
    total_fluid_target_l: float
    blocks: list[FuelingBlock]
    carb_deficit_g: float
    warnings: list[str] = field(default_factory=list)
