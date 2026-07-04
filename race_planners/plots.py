from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_course_profile(trackpoints: list[Any], aid_distances_km: list[float]) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    distances_km = [point.distance_from_start / 1000 for point in trackpoints]
    elevations = [point.elevation for point in trackpoints]
    ax.fill_between(distances_km, elevations, color="#2E86AB", alpha=0.28)
    ax.plot(distances_km, elevations, color="#2E86AB", linewidth=2.2)
    for aid_distance_km in aid_distances_km:
        ax.axvline(aid_distance_km, color="#E94F37", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Course Elevation Profile", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.18)
    plt.tight_layout()
    return fig


def plot_pace_profile(result: Any) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    kms = [split.km for split in result.splits]
    paces = [split.actual_pace_min_km for split in result.splits]
    colors = ["#E94F37" if split.grade_percent > 1.5 else "#2E86AB" for split in result.splits]
    ax.bar(kms, paces, width=0.85, color=colors, alpha=0.75)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pace (min/km)")
    ax.set_title("Planned Pace Profile", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    return fig
