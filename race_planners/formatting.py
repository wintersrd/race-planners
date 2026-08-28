from __future__ import annotations

from datetime import datetime, timedelta

from race_planners.models import CuratedEvent


def format_pace_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "-"
    total_seconds = max(int(round(minutes * 60)), 0)
    mins, secs = divmod(total_seconds, 60)
    return f"{mins}:{secs:02d}"


def format_duration_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "-"
    total_seconds = max(int(round(minutes * 60)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def format_signed_duration_minutes(minutes: float | None) -> str:
    """Format a signed duration like ``+0:45`` or ``-0:12`` (minus sign)."""
    if minutes is None:
        return "-"
    sign = "−" if minutes < 0 else "+"
    total_seconds = abs(int(round(minutes * 60)))
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{sign}{hours}:{mins:02d}:{secs:02d}"
    return f"{sign}{mins}:{secs:02d}"


def start_datetime(event: CuratedEvent) -> datetime | None:
    if event.start_time_local is None:
        return None
    return datetime.strptime(event.start_time_local, "%H:%M")


def format_clock_time(
    event: CuratedEvent, elapsed_minutes: float | None, locale: str = "en"
) -> str:
    if elapsed_minutes is None:
        return "-"
    start_dt = start_datetime(event)
    if start_dt is None:
        return "-"
    clock_dt = start_dt + timedelta(minutes=elapsed_minutes)

    if locale == "fr":
        return f"{clock_dt.hour}:{clock_dt.minute:02d}"

    hour_12 = clock_dt.hour % 12 or 12
    suffix = "AM" if clock_dt.hour < 12 else "PM"
    return f"{hour_12}:{clock_dt.minute:02d} {suffix}"
