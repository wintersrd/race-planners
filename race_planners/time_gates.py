"""Time gate (barrière horaire) matching and evaluation.

Gates are declared on curated events as absolute local clock times at
geographic locations. Station gates attach to the nearest aid station
within ``GATE_MATCH_TOLERANCE_KM``; finish gates (``distance_km is None``)
are evaluated against the projected total finish time.
"""

from __future__ import annotations

from dataclasses import replace

from race_planners.models import AidStation, GateCheck, TimeGate

GATE_MATCH_TOLERANCE_KM = 2.0
TIGHT_GATE_BUFFER_MIN = 30.0

_MINUTES_PER_DAY = 24 * 60


def _parse_clock_minutes(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid clock time {value!r}: expected HH:MM")
    try:
        hours, minutes = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid clock time {value!r}: expected HH:MM") from exc
    if not 0 <= hours <= 23 or not 0 <= minutes <= 59:
        raise ValueError(f"Invalid clock time {value!r}: out of range")
    return hours * 60 + minutes


def gate_deadline_elapsed_min(start_time_local: str, barrier_time_local: str) -> float:
    """Return the barrier as elapsed minutes from the event start.

    Barriers at or before the start clock time roll over to the next day
    (e.g. a 17:00 start with a 07:00 barrier gives 840 minutes).
    """
    elapsed = _parse_clock_minutes(barrier_time_local) - _parse_clock_minutes(start_time_local)
    if elapsed <= 0:
        elapsed += _MINUTES_PER_DAY
    return float(elapsed)


def attach_gates_to_stations(
    aid_stations: list[AidStation], time_gates: list[TimeGate]
) -> tuple[list[AidStation], list[TimeGate]]:
    """Attach station gates to the nearest aid station within tolerance.

    Returns the rebuilt station list (one gate per station at most, original
    ``TimeGate`` instances preserved for identity checks) plus the station
    gates that could not be matched. Finish gates are ignored here.
    """
    stations = list(aid_stations)
    unmatched: list[TimeGate] = []
    claimed: set[int] = set()
    for gate in time_gates:
        if gate.distance_km is None:
            continue
        best_index: int | None = None
        best_gap = GATE_MATCH_TOLERANCE_KM
        for index, station in enumerate(stations):
            if index in claimed:
                continue
            gap = abs(station.distance_km - gate.distance_km)
            if gap <= best_gap:
                best_index, best_gap = index, gap
        if best_index is None:
            unmatched.append(gate)
            continue
        claimed.add(best_index)
        stations[best_index] = replace(stations[best_index], time_gate=gate)
    return stations, unmatched


def evaluate_time_gates(
    arrivals: list[tuple[TimeGate, float]], start_time_local: str
) -> list[GateCheck]:
    """Build gate checks from (gate, projected arrival elapsed minutes) pairs.

    Arrival times must include rest accumulated at prior aid stations so the
    buffer reflects the clock time the runner actually passes the gate.
    Deadlines are forced to increase along the course: on races longer than
    24 hours a barrier clock time recurs daily, so a later gate whose base
    elapsed time does not exceed the previous deadline rolls forward whole
    days until it does (e.g. GRF166 starts 17:00; the 20:00 barrier at
    km 146.9 is the following day's 20:00, i.e. 1620 elapsed minutes).
    """
    checks: list[GateCheck] = []
    previous_deadline = 0.0
    for gate, arrival_elapsed_min in arrivals:
        deadline = gate_deadline_elapsed_min(start_time_local, gate.barrier_time_local)
        while deadline <= previous_deadline:
            deadline += _MINUTES_PER_DAY
        previous_deadline = deadline
        buffer_min = deadline - arrival_elapsed_min
        checks.append(
            GateCheck(
                label=gate.label,
                distance_km=gate.distance_km,
                barrier_time_local=gate.barrier_time_local,
                deadline_elapsed_min=deadline,
                arrival_elapsed_min=arrival_elapsed_min,
                buffer_min=buffer_min,
                missed=buffer_min < 0,
            )
        )
    return checks


def gate_warning_codes(checks: list[GateCheck]) -> list[str]:
    """Return domain warning message codes for missed and tight gates."""
    codes: list[str] = []
    for check in checks:
        if check.missed:
            codes.append(f"warning.gate_missed|label={check.label},buffer={check.buffer_min:.0f}")
        elif check.buffer_min < TIGHT_GATE_BUFFER_MIN:
            codes.append(f"warning.gate_tight|label={check.label},buffer={check.buffer_min:.0f}")
    return codes
