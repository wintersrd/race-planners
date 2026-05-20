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


@dataclass
class Course:
    course_id: str
    name: str
    gpx_path: Path
    aid_stops_km: list[float] = field(default_factory=list)
    terrain: str = "road"


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
    rpe_target: float | None = None
    hr_cap: int | None = None


@dataclass
class PaceSplit:
    km: float
    actual_pace_min_km: float
    grade_percent: float
    segment_time_min: float
    cumulative_time_min: float


@dataclass
class PlanResult:
    splits: list[PaceSplit]
    aid_arrival_times_min: list[float]
    total_time_min: float
    total_distance_km: float
