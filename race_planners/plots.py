from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from race_planners.i18n import t
from race_planners.models import PlanResult


def _pace_color(pace: float, median_pace: float) -> tuple[float, float, float]:
    """Return an RGB color for a pace based on how fast/slow it is vs median."""
    if median_pace <= 0:
        return (0.6, 0.6, 0.6)
    ratio = pace / median_pace
    if ratio < 0.92:
        return (0.2, 0.7, 0.3)
    if ratio < 0.98:
        return (0.4, 0.8, 0.4)
    if ratio <= 1.02:
        return (0.9, 0.8, 0.2)
    if ratio <= 1.10:
        return (0.95, 0.6, 0.2)
    return (0.85, 0.25, 0.2)


def plot_course_profile(
    trackpoints: list[Any], aid_distances_km: list[float], locale: str = "en"
) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    distances_km = np.array([point.distance_from_start / 1000 for point in trackpoints])
    elevations = np.array([point.elevation for point in trackpoints])

    ax.fill_between(distances_km, elevations, min(elevations) - 5, color="#A8D5BA", alpha=0.35)
    ax.plot(distances_km, elevations, color="#1B7340", linewidth=1.8)

    grades = np.gradient(elevations, distances_km * 1000) * 100
    for i in range(len(distances_km) - 1):
        if grades[i] > 3:
            ax.fill_between(
                distances_km[i : i + 2],
                elevations[i : i + 2],
                min(elevations) - 5,
                color="#E8B87D",
                alpha=0.3,
            )
        elif grades[i] < -3:
            ax.fill_between(
                distances_km[i : i + 2],
                elevations[i : i + 2],
                min(elevations) - 5,
                color="#7DB8E8",
                alpha=0.2,
            )

    for aid_km in aid_distances_km:
        idx = np.searchsorted(distances_km, aid_km)
        elev_at_aid = elevations[idx] if idx < len(elevations) else elevations[-1]
        ax.plot(aid_km, elev_at_aid, "v", color="#E94F37", markersize=8, zorder=5)
        ax.axvline(aid_km, color="#E94F37", linestyle=":", linewidth=0.8, alpha=0.4)

    ax.set_xlabel(t("overview.distance", locale) + " (km)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    return fig


def plot_pace_profile(result: PlanResult, locale: str = "en") -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    kms = [split.km for split in result.splits]
    paces = [split.actual_pace_min_km for split in result.splits]
    median_pace = float(np.median(paces)) if paces else 6.0

    colors = [_pace_color(p, median_pace) for p in paces]
    ax.bar(kms, paces, width=0.85, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.axhline(median_pace, color="#555555", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(
        kms[-1] * 0.98,
        median_pace,
        f"Median {median_pace:.1f}",
        fontsize=8,
        ha="right",
        va="bottom",
        color="#555555",
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=(0.2, 0.7, 0.3), alpha=0.7, label=t("pace_band.fast", locale)),
        Patch(facecolor=(0.9, 0.8, 0.2), alpha=0.7, label=t("pace_band.on_target", locale)),
        Patch(facecolor=(0.85, 0.25, 0.2), alpha=0.7, label=t("pace_band.costly", locale)),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel(t("overview.distance", locale) + " (km)", fontsize=11)
    ax.set_ylabel(t("col.pace", locale) + " (min/km)", fontsize=11)
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    return fig


def plot_cumulative_time(result: PlanResult, event: Any, locale: str = "en") -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    kms = [0.0]
    times = [0.0]
    cumulative = 0.0
    for split in result.splits:
        cumulative = split.cumulative_time_min
        kms.append(split.km)
        times.append(cumulative)

    ax.plot(kms, times, color="#2E86AB", linewidth=2.5)
    ax.fill_between(kms, times, color="#2E86AB", alpha=0.15)

    for aid_eta in result.aid_station_etas:
        ax.plot(
            aid_eta.distance_km,
            aid_eta.arrival_elapsed_time_min,
            "v",
            color="#E94F37",
            markersize=7,
            zorder=5,
        )

    ax.set_xlabel(t("overview.distance", locale) + " (km)", fontsize=11)
    ax.set_ylabel(t("metric.elapsed", locale) + " (min)", fontsize=11)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    return fig


def plot_terrain_breakdown(result: PlanResult, locale: str = "en") -> Figure:
    from race_planners.i18n import t

    climb_dist = 0.0
    flat_dist = 0.0
    descent_dist = 0.0
    climb_time = 0.0
    flat_time = 0.0
    descent_time = 0.0

    prev_km = 0.0
    for split in result.splits:
        seg_dist = split.km - prev_km
        seg_time = split.segment_time_min
        if split.grade_percent > 2.0:
            climb_dist += seg_dist
            climb_time += seg_time
        elif split.grade_percent < -2.0:
            descent_dist += seg_dist
            descent_time += seg_time
        else:
            flat_dist += seg_dist
            flat_time += seg_time
        prev_km = split.km

    labels = [
        t("split.climb", locale),
        t("split.flat", locale),
        t("split.descent", locale),
    ]
    distances = [climb_dist, flat_dist, descent_dist]
    times = [climb_time, flat_time, descent_time]
    colors = ["#E8B87D", "#A8D5BA", "#7DB8E8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.bar(labels, distances, color=colors, alpha=0.8)
    ax1.set_ylabel("Distance (km)", fontsize=10)
    ax1.tick_params(labelsize=9)

    ax2.bar(labels, times, color=colors, alpha=0.8)
    ax2.set_ylabel("Time (min)", fontsize=10)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    return fig


def plot_half_comparison(result: PlanResult, locale: str = "en") -> Figure:
    """Bar chart comparing first-half vs second-half time and pace."""
    total_km = result.total_distance_km
    half_km = total_km / 2.0

    first_time = 0.0
    first_dist = 0.0
    second_time = 0.0
    second_dist = 0.0

    for split in result.splits:
        if split.km <= half_km:
            first_time += split.segment_time_min
            first_dist += split.km - (first_dist)
        else:
            second_time += split.segment_time_min
            second_dist += split.km - (first_dist + second_dist)

    first_pace = first_time / first_dist if first_dist > 0 else 0
    second_pace = second_time / second_dist if second_dist > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    labels = [t("split.first_half", locale), t("split.second_half", locale)]
    ax1.bar(labels, [first_time, second_time], color=["#2E86AB", "#E94F37"], alpha=0.8)
    ax1.set_ylabel(t("split.time", locale) + " (min)", fontsize=10)

    ax2.bar(labels, [first_pace, second_pace], color=["#2E86AB", "#E94F37"], alpha=0.8)
    ax2.set_ylabel(t("split.pace", locale) + " (min/km)", fontsize=10)

    plt.tight_layout()
    return fig


def plot_temperature_curve(
    peak_temp_c: float,
    start_time_local: str | None,
    total_time_min: float,
    locale: str = "en",
) -> Figure:
    from race_planners.weather import temperature_at_elapsed

    hours = np.linspace(0, total_time_min / 60.0, max(int(total_time_min / 10), 20))
    temps = [temperature_at_elapsed(peak_temp_c, hr, start_time_local) for hr in hours]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(hours, temps, color="#E94F37", linewidth=2.5)
    ax.fill_between(hours, temps, min(temps) - 2, color="#E94F37", alpha=0.12)
    ax.axhline(peak_temp_c, color="#555555", linestyle="--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel(t("metric.elapsed", locale) + " (h)", fontsize=11)
    ax.set_ylabel("°C", fontsize=11)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    return fig


def plot_heat_impact(
    peak_temp_c: float,
    start_time_local: str | None,
    total_time_min: float,
    splits: list[Any],
    locale: str = "en",
) -> Figure:
    from race_planners.weather import heat_multiplier, temperature_at_elapsed

    kms = [split.km for split in splits]
    penalties: list[float] = []

    cumulative_time = 0.0
    for split in splits:
        elapsed_hr = cumulative_time / 60.0
        temp_c = temperature_at_elapsed(peak_temp_c, elapsed_hr, start_time_local)
        penalty = (heat_multiplier(temp_c) - 1.0) * 100.0
        penalties.append(max(penalty, 0.0))
        cumulative_time = split.cumulative_time_min

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(kms, penalties, width=0.85, color="#E94F37", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(t("overview.distance", locale) + " (km)", fontsize=11)
    ax.set_ylabel("%", fontsize=11)
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    return fig
