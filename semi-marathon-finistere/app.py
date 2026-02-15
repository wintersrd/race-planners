"""
Semi-Marathon du Finistère Race Pacing App
Streamlit version of the Jupyter notebook race planner

Run with: streamlit run app.py
"""

import streamlit as st
import xml.etree.ElementTree as ET
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Half Marathon Race Pacing",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR TABS
# =============================================================================
st.markdown("""
<style>
    /* Make tab headers larger and more pronounced */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px 10px 0px 10px;
        border-radius: 10px 10px 0px 0px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        padding: 15px 25px;
        background-color: #e8eaed;
        border-radius: 8px 8px 0px 0px;
        border: 2px solid transparent;
        transition: all 0.2s ease;
        color: #555;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #dce3ea;
        color: #2E86AB;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border: 2px solid #2E86AB !important;
        border-bottom: 2px solid white !important;
        color: #2E86AB !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        padding: 20px;
        border: 2px solid #2E86AB;
        border-top: none;
        border-radius: 0px 0px 10px 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TRANSLATIONS DICTIONARY - English/French
# =============================================================================

TRANSLATIONS = {
    'en': {
        'title': '🏃‍♀️ Half Marathon Race Pacing',
        'subtitle': 'Semi-Marathon du Finistère',
        'course': 'Course',
        'rest_stops_at': 'Rest stops at',
        'set_your_target': '🎯 Set Your Target',
        'target_finish_time': 'Target Finish Time',
        'target_avg_pace': 'Target Average Pace',
        'finish_time': 'Finish Time:',
        'pace': 'Pace:',
        'power_fade': 'Power Fade:',
        'rest_duration': 'Rest Duration:',
        'calculate': 'Calculate Pacing',
        'elapsed': 'Elapsed',
        'split_time': 'Split Time',
        'actual': 'Actual',
        'elev_change': 'Elev Δ',
        'rest': 'Rest',
        'finish': 'Finish',
        'km': 'km',
        'grade': 'Grade',
        'total': 'Total',
        'strategy_even': 'EVEN PACING',
        'strategy_positive': 'POSITIVE SPLIT',
        'strategy_negative': 'NEGATIVE SPLIT',
        'running_time': 'Running time',
        'rest_time': 'Rest time',
        'total_time': 'Total time',
        'avg_pace': 'Avg pace',
        'gap_pace': 'GAP pace',
        'split_analysis': 'Split Analysis',
        'first_half': '1st half',
        'second_half': '2nd half',
        'elevation': 'Elevation',
        'uphill_kms': 'Uphill kms',
        'downhill_kms': 'Downhill kms',
        'split': 'Split',
        'slower_2nd': 'slower 2nd half',
        'faster_2nd': 'faster 2nd half',
        'even': 'Even',
        'rest_stop_arrival': '🚰 Rest Stop Arrival Times',
        'km_splits': '📊 Kilometer Splits',
        'course_profile': 'Course Elevation Profile',
        'pace_per_km': 'Pace per Kilometer',
        'range': 'Range',
        'uphill': 'uphill (slower)',
        'downhill': 'downhill (faster)',
        'faster_start': 'faster start',
        'slower_finish': 'slower finish',
        'slower_start': 'slower start',
        'faster_finish': 'faster finish',
        'seconds_per_stop': 'Seconds per rest stop',
        'no_stops': 'no stops',
        'quick_ref': '📋 Quick Reference',
        'level': 'Level',
        'elite': 'Elite',
        'advanced': 'Advanced',
        'competitive': 'Competitive',
        'strong': 'Strong',
        'intermediate': 'Intermediate',
        'recreational': 'Recreational',
        'target': 'TARGET',
        'strategy': 'Strategy',
        'pace_adjustment': 'Pace adjustment',
        'stops': 'stops',
        'print_pocket': '🖨️ Pocket Card',
        'print_wrist': '🖨️ Wrist Band',
        'course_sections': '📍 Course Sections',
        'section': 'Section',
        'distance': 'Distance',
        'time': 'Time',
        'elev': 'Elev',
        'negative_faster_finish': 'Negative = stronger finish',
        'positive_faster_start': 'Positive = faster start',
        'zero_even_pace': '0 = even pace',
        'input_mode': 'Input Mode:',
        'pacing_tip': 'Pacing Tip',
        'summary': '📊 Summary',
        'tables': '📋 Tables',
        'print_export': '🖨️ Print/Export',
        'download_pocket': 'Download Pocket Card',
        'download_wrist': 'Download Wrist Band',
        'how_to_use': 'How to Use',
        'instructions': '''1. **Set your target** - Enter your goal finish time or target pace
2. **Adjust power fade** - Use negative values for a stronger finish, positive for a faster start
3. **Set rest duration** - How long you plan to stop at each water station
4. **Click "Calculate Pacing"** - View your personalized pacing plan
5. **Print or download** - Use the Print/Export tab for race-day reference''',
    },
    'fr': {
        'title': '🏃‍♀️ Allure Semi-Marathon',
        'subtitle': 'Semi-Marathon du Finistère',
        'course': 'Parcours',
        'rest_stops_at': 'Ravitaillements à',
        'set_your_target': '🎯 Définissez votre objectif',
        'target_finish_time': 'Temps cible',
        'target_avg_pace': 'Allure moyenne cible',
        'finish_time': 'Temps:',
        'pace': 'Allure:',
        'power_fade': 'Gestion effort:',
        'rest_duration': 'Durée ravito:',
        'calculate': 'Calculer',
        'elapsed': 'Cumul',
        'split_time': 'Intervalle',
        'actual': 'Réelle',
        'elev_change': 'Déniv.',
        'rest': 'Ravito',
        'finish': 'Arrivée',
        'km': 'km',
        'grade': 'Pente',
        'total': 'Total',
        'strategy_even': 'ALLURE CONSTANTE',
        'strategy_positive': 'SPLIT POSITIF',
        'strategy_negative': 'SPLIT NÉGATIF',
        'running_time': 'Temps de course',
        'rest_time': 'Temps ravito',
        'total_time': 'Temps total',
        'avg_pace': 'Allure moy',
        'gap_pace': 'Allure GAP',
        'split_analysis': 'Analyse des splits',
        'first_half': '1ère moitié',
        'second_half': '2ème moitié',
        'elevation': 'Dénivelé',
        'uphill_kms': 'kms montée',
        'downhill_kms': 'kms descente',
        'split': 'Split',
        'slower_2nd': '2ème moitié plus lente',
        'faster_2nd': '2ème moitié plus rapide',
        'even': 'Égal',
        'rest_stop_arrival': '🚰 Temps aux Ravitaillements',
        'km_splits': '📊 Splits par Kilomètre',
        'course_profile': 'Profil Altimétrique',
        'pace_per_km': 'Allure par Kilomètre',
        'range': 'Plage',
        'uphill': 'montée (plus lent)',
        'downhill': 'descente (plus rapide)',
        'faster_start': 'départ rapide',
        'slower_finish': 'fin plus lente',
        'slower_start': 'départ lent',
        'faster_finish': 'fin plus rapide',
        'seconds_per_stop': 'Secondes par ravitaillement',
        'no_stops': 'sans arrêt',
        'quick_ref': '📋 Référence Rapide',
        'level': 'Niveau',
        'elite': 'Élite',
        'advanced': 'Avancé',
        'competitive': 'Compétitif',
        'strong': 'Confirmé',
        'intermediate': 'Intermédiaire',
        'recreational': 'Loisir',
        'target': 'OBJECTIF',
        'strategy': 'Stratégie',
        'pace_adjustment': 'Ajustement allure',
        'stops': 'arrêts',
        'print_pocket': '🖨️ Carte Poche',
        'print_wrist': '🖨️ Brassard',
        'course_sections': '📍 Sections du Parcours',
        'section': 'Section',
        'distance': 'Distance',
        'time': 'Temps',
        'elev': 'Déniv.',
        'negative_faster_finish': 'Négatif = fin plus rapide',
        'positive_faster_start': 'Positif = départ rapide',
        'zero_even_pace': '0 = allure constante',
        'input_mode': 'Mode saisie:',
        'pacing_tip': "Conseil d'Allure",
        'summary': '📊 Résumé',
        'tables': '📋 Tableaux',
        'print_export': '🖨️ Imprimer/Exporter',
        'download_pocket': 'Télécharger Carte Poche',
        'download_wrist': 'Télécharger Brassard',
        'how_to_use': 'Comment Utiliser',
        'instructions': '''1. **Définissez votre objectif** - Entrez votre temps cible ou allure cible
2. **Ajustez la gestion d'effort** - Valeurs négatives pour une fin plus forte, positives pour un départ rapide
3. **Définissez la durée des ravitos** - Combien de temps vous prévoyez vous arrêter à chaque station
4. **Cliquez "Calculer"** - Consultez votre plan d'allure personnalisé
5. **Imprimez ou téléchargez** - Utilisez l'onglet Imprimer/Exporter pour votre aide-mémoire''',
    }
}

def t(key: str, lang: str) -> str:
    """Get translated string for the specified language."""
    return TRANSLATIONS.get(lang, {}).get(key, key)

# =============================================================================
# COURSE SECTIONS - Based on actual GPX elevation analysis
# =============================================================================

COURSE_SECTIONS = [
    {
        'start_km': 0,
        'end_km': 5.3,
        'name_en': 'The Opening Climb',
        'name_fr': "L'Ascension d'Ouverture",
        'strategy_en': 'Main climb (0-3km +55m), then recover on descent to rest stop',
        'strategy_fr': 'Montée principale (0-3km +55m), récupérez dans la descente',
        'pacing_en': "DON'T BANK TIME HERE. Accept slower pace on the climb (your GAP will be on target). Let the descent come naturally - don't sprint it. Arrive at rest 1 feeling controlled, not spent.",
        'pacing_fr': "NE CHERCHEZ PAS À GAGNER DU TEMPS ICI. Acceptez un rythme plus lent dans la montée (votre GAP sera bon). Laissez la descente venir naturellement. Arrivez au ravito 1 maîtrisé, pas épuisé.",
        'icon': '⛰️'
    },
    {
        'start_km': 5.3,
        'end_km': 9.1,
        'name_en': 'The Rolling Ascent',
        'name_fr': "L'Ascension Roulante",
        'strategy_en': 'Steady climb with variation (+39m gain), conserve energy',
        'strategy_fr': 'Montée régulière avec variations (+39m), économisez',
        'pacing_en': "This is THE CRITICAL SECTION. You'll be tempted to push after rest 1, but there's still 16km to go. Keep effort steady on the rollers. If you can't talk comfortably here, you're going too hard.",
        'pacing_fr': "C'est LA SECTION CRITIQUE. Vous serez tenté d'accélérer après le ravito 1, mais il reste 16km. Gardez un effort constant. Si vous ne pouvez pas parler confortablement, vous allez trop vite.",
        'icon': '📈'
    },
    {
        'start_km': 9.1,
        'end_km': 14.5,
        'name_en': 'The Big Drop & Climb',
        'name_fr': 'La Grande Descente et Montée',
        'strategy_en': 'Enjoy the big descent (-52m), then start the second climb to rest',
        'strategy_fr': 'Profitez de la grande descente (-52m), puis attaquez la seconde montée',
        'pacing_en': "Use the big descent for FREE SPEED but stay RELAXED - don't burn matches. When the climb starts at ~12.5km, dig in mentally. Rest stop 3 at 14.5km is MID-CLIMB - don't stop too long or you'll get cold legs.",
        'pacing_fr': "Profitez de la grande descente pour de la VITESSE GRATUITE mais restez DÉTENDU. Quand la montée commence vers 12.5km, accrochez-vous mentalement. Le ravito 3 à 14.5km est EN PLEINE MONTÉE - pas de pause trop longue.",
        'icon': '📉'
    },
    {
        'start_km': 14.5,
        'end_km': 21.1,
        'name_en': 'The Final Push',
        'name_fr': 'La Poussée Finale',
        'strategy_en': 'Finish the climb, then descend (-39m) and sprint to the finish',
        'strategy_fr': 'Terminez la montée, descendez (-39m) et sprintez vers l\'arrivée',
        'pacing_en': "You're past the worst! Finish the remaining ~2km of climb, then GRAVITY IS YOUR FRIEND on the -39m descent. Open up the stride. Last 2km is flat/rolling - leave everything on the course. This is what you trained for!",
        'pacing_fr': "Le pire est passé! Finissez les ~2km de montée restants, puis la GRAVITÉ EST VOTRE AMIE dans la descente de -39m. Ouvrez la foulée. Les derniers 2km sont plats - donnez tout. C'est pour ça que vous vous êtes entraîné!",
        'icon': '🏁'
    }
]

REST_STOPS = [5.3, 9.1, 14.5]
# Use absolute path based on script location for deployment compatibility
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPX_FILE = os.path.join(SCRIPT_DIR, 'WR-GPX-Semi-marathon-du-Finistere.gpx')
SMOOTHING_WINDOW = 5

# =============================================================================
# DATA CLASSES AND CORE FUNCTIONS
# =============================================================================

@dataclass
class TrackPoint:
    lat: float
    lon: float
    elevation: float
    time: str
    distance_from_start: float = 0.0
    grade_percent: float = 0.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters."""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_gpx(filepath: str) -> List[TrackPoint]:
    """Parse GPX file and return list of TrackPoints."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    trackpoints = []
    cumulative_distance = 0.0
    prev_point = None
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        ele_elem = trkpt.find('gpx:ele', ns)
        elevation = float(ele_elem.text) if ele_elem is not None else 0.0
        time_elem = trkpt.find('gpx:time', ns)
        time = time_elem.text if time_elem is not None else ''
        if prev_point is not None:
            distance = haversine(prev_point.lat, prev_point.lon, lat, lon)
            cumulative_distance += distance
        point = TrackPoint(lat=lat, lon=lon, elevation=elevation, time=time, distance_from_start=cumulative_distance)
        trackpoints.append(point)
        prev_point = point
    return trackpoints


def smooth_elevation(trackpoints: List[TrackPoint], window_size: int = 3) -> List[TrackPoint]:
    """Smooth elevation data using moving average."""
    if len(trackpoints) < window_size:
        return trackpoints
    smoothed = []
    half_window = window_size // 2
    for i, point in enumerate(trackpoints):
        start_idx = max(0, i - half_window)
        end_idx = min(len(trackpoints), i + half_window + 1)
        avg_elevation = sum(trackpoints[j].elevation for j in range(start_idx, end_idx)) / (end_idx - start_idx)
        smoothed_point = TrackPoint(lat=point.lat, lon=point.lon, elevation=avg_elevation, time=point.time, distance_from_start=point.distance_from_start)
        smoothed.append(smoothed_point)
    return smoothed


def calculate_segment_grades(trackpoints: List[TrackPoint], segment_size_m: float = 100) -> List[TrackPoint]:
    """Calculate grade percentage for each trackpoint."""
    if len(trackpoints) < 2:
        return trackpoints
    for i in range(1, len(trackpoints)):
        distance_diff = trackpoints[i].distance_from_start - trackpoints[i-1].distance_from_start
        elevation_diff = trackpoints[i].elevation - trackpoints[i-1].elevation
        if distance_diff > 0:
            grade = (elevation_diff / distance_diff) * 100
        else:
            grade = 0.0
        trackpoints[i].grade_percent = grade
    window = 5
    grades = [p.grade_percent for p in trackpoints]
    for i in range(len(trackpoints)):
        start = max(0, i - window // 2)
        end = min(len(grades), i + window // 2 + 1)
        trackpoints[i].grade_percent = sum(grades[start:end]) / (end - start)
    return trackpoints


def gap_factor(grade_percent: float) -> float:
    """Calculate GAP multiplier based on grade."""
    return 0.0021 * grade_percent**2 + 0.034 * grade_percent + 1


def calculate_gap_adjusted_distance(trackpoints: List[TrackPoint]) -> float:
    """Calculate total GAP-adjusted distance in meters."""
    total_gap_distance = 0.0
    for i in range(1, len(trackpoints)):
        segment_distance = trackpoints[i].distance_from_start - trackpoints[i-1].distance_from_start
        avg_grade = (trackpoints[i].grade_percent + trackpoints[i-1].grade_percent) / 2
        gap_mult = gap_factor(avg_grade)
        total_gap_distance += segment_distance * gap_mult
    return total_gap_distance


def format_time(minutes: float) -> str:
    """Format minutes as HH:MM:SS or MM:SS."""
    total_seconds = int(minutes * 60)
    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    mins = remaining // 60
    secs = remaining % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def parse_time_input(time_str: str) -> float:
    """Parse time string (HH:MM:SS or MM:SS) to minutes."""
    parts = time_str.strip().split(':')
    if len(parts) == 2:
        return float(parts[0]) + float(parts[1]) / 60
    elif len(parts) == 3:
        return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def parse_pace_input(pace_str: str) -> float:
    """Parse pace string (MM:SS) to minutes per km."""
    parts = pace_str.strip().split(':')
    if len(parts) == 2:
        return float(parts[0]) + float(parts[1]) / 60
    else:
        raise ValueError(f"Invalid pace format: {pace_str}")


def get_fade_multiplier(power_fade: float, km: float, total_distance_km: float) -> float:
    """Calculate pace multiplier based on power fade setting."""
    if power_fade == 0:
        return 1.0
    halfway = total_distance_km / 2
    fade_factor = power_fade * 0.005
    if km <= halfway:
        return 1.0 - fade_factor
    else:
        return 1.0 + fade_factor


def calculate_elevation_changes(trackpoints: List[TrackPoint], start_km: float, end_km: float) -> Tuple[float, float]:
    """Calculate elevation gain and loss between two distances."""
    start_m = start_km * 1000
    end_m = end_km * 1000
    segment_points = [p for p in trackpoints if start_m <= p.distance_from_start <= end_m]

    if len(segment_points) < 2:
        return 0, 0

    gain = 0
    loss = 0
    for i in range(1, len(segment_points)):
        diff = segment_points[i].elevation - segment_points[i-1].elevation
        if diff > 0:
            gain += diff
        else:
            loss += abs(diff)

    return gain, loss


def calculate_segment_gap_factor(trackpoints: List[TrackPoint], start_m: float, end_m: float) -> Tuple[float, float, float]:
    """Calculate weighted average GAP factor for a segment."""
    total_gap_weighted = 0.0
    total_distance = 0.0
    grades_in_segment = []
    elevations_in_segment = []

    for i in range(1, len(trackpoints)):
        pt_start = trackpoints[i-1].distance_from_start
        pt_end = trackpoints[i].distance_from_start

        if pt_end <= start_m or pt_start >= end_m:
            continue

        overlap_start = max(pt_start, start_m)
        overlap_end = min(pt_end, end_m)
        overlap_distance = overlap_end - overlap_start

        if overlap_distance > 0:
            avg_grade = (trackpoints[i].grade_percent + trackpoints[i-1].grade_percent) / 2
            gap_mult = gap_factor(avg_grade)
            total_gap_weighted += overlap_distance * gap_mult
            total_distance += overlap_distance
            grades_in_segment.append(avg_grade)
            elevations_in_segment.append(trackpoints[i].elevation)

    if total_distance > 0:
        weighted_gap_mult = total_gap_weighted / total_distance
        avg_grade = sum(grades_in_segment) / len(grades_in_segment) if grades_in_segment else 0
        avg_elevation = sum(elevations_in_segment) / len(elevations_in_segment) if elevations_in_segment else 0
    else:
        weighted_gap_mult = 1.0
        avg_grade = 0.0
        avg_elevation = 0.0

    return weighted_gap_mult, avg_grade, avg_elevation


def calculate_pacing(trackpoints: List[TrackPoint], target_finish_time_min: float,
                     rest_stops: List[float], total_distance_km: float,
                     gap_adjusted_distance_m: float, power_fade: float = 0.0,
                     rest_duration_sec: int = 30) -> dict:
    """Calculate pacing based on GAP model."""
    gap_adjusted_distance_km = gap_adjusted_distance_m / 1000
    rest_duration_min = rest_duration_sec / 60.0

    total_rest_time_min = len(rest_stops) * rest_duration_min
    running_time_min = target_finish_time_min - total_rest_time_min

    if running_time_min <= 0:
        raise ValueError(f"Rest time exceeds target finish time")

    base_gap_pace = running_time_min / gap_adjusted_distance_km

    km_splits = []
    cumulative_time = 0.0

    for km in range(1, int(total_distance_km) + 1):
        start_dist = (km - 1) * 1000
        end_dist = km * 1000

        gap_mult, avg_grade, avg_elevation = calculate_segment_gap_factor(trackpoints, start_dist, end_dist)
        fade_mult = get_fade_multiplier(power_fade, km, total_distance_km)
        gap_pace = base_gap_pace * fade_mult
        actual_pace = gap_pace * gap_mult

        segment_time = actual_pace
        cumulative_time += segment_time

        km_splits.append({
            'km': km,
            'actual_pace_min_km': actual_pace,
            'gap_pace_min_km': gap_pace,
            'base_gap_pace': base_gap_pace,
            'gap_mult': gap_mult,
            'fade_mult': fade_mult,
            'grade_percent': avg_grade,
            'elevation_m': avg_elevation,
            'segment_time_min': segment_time,
            'cumulative_time_min': cumulative_time
        })

    # Handle final partial km
    final_km = int(total_distance_km)
    remaining_distance_km = total_distance_km - final_km
    if remaining_distance_km > 0.01:
        start_dist = final_km * 1000
        end_dist = total_distance_km * 1000

        gap_mult, avg_grade, avg_elevation = calculate_segment_gap_factor(trackpoints, start_dist, end_dist)
        fade_mult = get_fade_multiplier(power_fade, total_distance_km, total_distance_km)
        gap_pace = base_gap_pace * fade_mult
        actual_pace = gap_pace * gap_mult
        segment_time = actual_pace * remaining_distance_km
        cumulative_time += segment_time
        km_splits.append({
            'km': round(total_distance_km, 2),
            'actual_pace_min_km': actual_pace,
            'gap_pace_min_km': gap_pace,
            'base_gap_pace': base_gap_pace,
            'gap_mult': gap_mult,
            'fade_mult': fade_mult,
            'grade_percent': avg_grade,
            'elevation_m': avg_elevation,
            'segment_time_min': segment_time,
            'cumulative_time_min': cumulative_time
        })

    # Calculate rest stop data
    rest_stop_data = []
    prev_arrival_time = 0.0
    prev_distance = 0.0

    for stop_km in rest_stops:
        stop_time = 0.0
        for split in km_splits:
            if split['km'] >= stop_km:
                fraction = stop_km - int(stop_km)
                if fraction > 0:
                    stop_time = split['cumulative_time_min'] - split['segment_time_min'] * (1 - fraction)
                else:
                    stop_time = split['cumulative_time_min']
                break
            stop_time = split['cumulative_time_min']

        split_time = stop_time - prev_arrival_time
        split_distance = stop_km - prev_distance

        elev_gain, elev_loss = calculate_elevation_changes(trackpoints, prev_distance, stop_km)

        if split_distance > 0:
            actual_pace_segment = split_time / split_distance
            segment_gap_pace = base_gap_pace * get_fade_multiplier(power_fade, (prev_distance + stop_km) / 2, total_distance_km)
        else:
            actual_pace_segment = 0
            segment_gap_pace = base_gap_pace

        rest_stop_data.append({
            'stop_number': len(rest_stop_data) + 1,
            'distance_km': stop_km,
            'arrival_time_min': stop_time,
            'elapsed_time_str': format_time(stop_time),
            'split_from_prev_min': split_time,
            'split_distance_km': split_distance,
            'actual_pace_min_km': actual_pace_segment,
            'gap_pace_min_km': segment_gap_pace,
            'elev_gain_m': elev_gain,
            'elev_loss_m': elev_loss,
            'suggested_rest_min': rest_duration_min,
            'departure_time_min': stop_time + rest_duration_min
        })
        prev_arrival_time = stop_time
        prev_distance = stop_km

    return {
        'km_splits': km_splits,
        'rest_stops': rest_stop_data,
        'target_gap_pace': base_gap_pace,
        'total_gap_distance_km': gap_adjusted_distance_km,
        'calculated_finish_time_min': cumulative_time,
        'power_fade': power_fade,
        'rest_duration_sec': rest_duration_sec,
        'total_rest_time_min': total_rest_time_min,
        'running_time_min': running_time_min
    }

# =============================================================================
# CACHED DATA LOADING
# =============================================================================

@st.cache_data
def load_gpx_data(gpx_file: str, smoothing_window: int):
    """Load and process GPX file once, cache results."""
    raw_trackpoints = parse_gpx(gpx_file)
    smoothed = smooth_elevation(raw_trackpoints, smoothing_window)
    trackpoints = calculate_segment_grades(smoothed)
    total_distance_m = trackpoints[-1].distance_from_start
    gap_distance_m = calculate_gap_adjusted_distance(trackpoints)
    return trackpoints, total_distance_m / 1000, gap_distance_m

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_elevation_profile(trackpoints: List[TrackPoint], rest_stops: List[float],
                           total_distance_km: float, lang: str) -> plt.Figure:
    """Create elevation profile plot."""
    fig, ax = plt.subplots(figsize=(12, 4))
    distances_km = [p.distance_from_start / 1000 for p in trackpoints]
    elevations = [p.elevation for p in trackpoints]
    ax.fill_between(distances_km, elevations, alpha=0.3, color='#2E86AB')
    ax.plot(distances_km, elevations, color='#2E86AB', linewidth=2)
    for stop in rest_stops:
        ax.axvline(x=stop, color='#E94F37', linestyle='--', linewidth=1.5, alpha=0.8)
        stop_idx = min(range(len(trackpoints)), key=lambda i: abs(trackpoints[i].distance_from_start / 1000 - stop))
        stop_elev = trackpoints[stop_idx].elevation
        ax.annotate(f'Rest {rest_stops.index(stop) + 1}\n{stop} km', xy=(stop, stop_elev),
                   xytext=(stop + 0.3, stop_elev + 8), fontsize=10, color='#E94F37', fontweight='bold')
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel(f'{t("elevation", lang)} (m)', fontsize=12)
    ax.set_title(t('course_profile', lang), fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_distance_km + 0.5)
    min_elev = min(elevations)
    max_elev = max(elevations)
    ax.text(0.02, 0.95, f'{t("range", lang)}: {min_elev:.0f}m - {max_elev:.0f}m', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    return fig


def plot_pace_comparison(pacing_data: dict, total_distance_km: float, lang: str) -> plt.Figure:
    """Create pace comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 4))
    km_splits = pacing_data['km_splits']
    kms = [s['km'] for s in km_splits]
    actual_paces = [s['actual_pace_min_km'] for s in km_splits]
    gap_pace = pacing_data['target_gap_pace']
    colors = ['#E94F37' if p > gap_pace else '#2E86AB' for p in actual_paces]
    ax.bar(kms, [p * 60 for p in actual_paces], width=0.8, color=colors, alpha=0.7, label='Actual Pace')
    ax.axhline(y=gap_pace * 60, color='#1B998B', linestyle='-', linewidth=2.5, label=f'GAP Pace ({format_time(gap_pace)}/km)')
    ax.axvline(x=total_distance_km / 2, color='#888888', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(t('km', lang).upper(), fontsize=12)
    ax.set_ylabel(f'{t("pace", lang)} (sec/km)', fontsize=12)
    ax.set_title(t('pace_per_km', lang), fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.text(0.02, 0.95, f'Red: {t("uphill", lang)} | Blue: {t("downhill", lang)}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    return fig

# =============================================================================
# HELPER FUNCTIONS FOR DISPLAY
# =============================================================================

def generate_pocket_card(pacing_data: dict, total_distance_km: float) -> str:
    """Generate pocket card text for download."""
    lines = [
        "Point   km      Time      Pace      Elev",
        "─" * 42,
    ]
    for stop in pacing_data['rest_stops']:
        lines.append(f"R{stop['stop_number']}      {stop['distance_km']:.1f}     {stop['elapsed_time_str']}      {format_time(stop.get('actual_pace_min_km', 0))}/km   +{stop.get('elev_gain_m', 0):.0f}/-{stop.get('elev_loss_m', 0):.0f}")
    lines.append(f"FINISH  {total_distance_km:.1f}    {format_time(pacing_data['calculated_finish_time_min'])}      -         -")
    lines.append("─" * 42)
    return "\n".join(lines)


def generate_wrist_band(pacing_data: dict, total_distance_km: float) -> str:
    """Generate wrist band text for download."""
    lines = [
        "km     Time    Pace",
        "─" * 22,
    ]
    for stop in pacing_data['rest_stops']:
        lines.append(f"{stop['distance_km']:.1f}    {stop['elapsed_time_str']}    {format_time(stop.get('actual_pace_min_km', 0))}")
    lines.append(f"{total_distance_km:.1f}  {format_time(pacing_data['calculated_finish_time_min'])}    -")
    lines.append("─" * 22)
    return "\n".join(lines)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load GPX data (cached)
    trackpoints, total_distance_km, gap_adjusted_distance_m = load_gpx_data(GPX_FILE, SMOOTHING_WINDOW)

    # Title and language selector
    col_title, col_lang = st.columns([4, 1])

    with col_lang:
        lang_options = [("English 🇬🇧", "en"), ("Français 🇫🇷", "fr")]
        lang_selection = st.selectbox(
            "Language",
            options=lang_options,
            format_func=lambda x: x[0],
            label_visibility="collapsed"
        )
        lang = lang_selection[1]

    with col_title:
        st.title(t('title', lang))
        st.subheader(t('subtitle', lang))

    # Course info
    st.markdown(f"**{t('course', lang)}:** 21.06 km with ~153m elevation gain")
    st.markdown(f"**{t('rest_stops_at', lang)}:** 5.3 km, 9.1 km, 14.5 km")

    # Instructions expander
    with st.expander(f"📖 {t('how_to_use', lang)}"):
        st.markdown(t('instructions', lang))

    # Sidebar for inputs
    with st.sidebar:
        st.markdown(f"### {t('set_your_target', lang)}")

        input_mode = st.radio(
            t('input_mode', lang),
            [t('target_finish_time', lang), t('target_avg_pace', lang)]
        )

        if input_mode == t('target_finish_time', lang):
            finish_time_str = st.text_input(t('finish_time', lang), value="01:45:00")
            avg_pace_str = "05:00"
        else:
            avg_pace_str = st.text_input(t('pace', lang), value="05:00")
            finish_time_str = "01:45:00"

        power_fade = st.slider(
            f"{t('power_fade', lang)} (-10 to +10)",
            min_value=-10, max_value=10, value=0, step=1
        )
        st.caption(f"💡 {t('negative_faster_finish', lang)} | {t('positive_faster_start', lang)} | {t('zero_even_pace', lang)}")

        rest_duration = st.slider(
            f"{t('rest_duration', lang)} (seconds)",
            min_value=0, max_value=120, value=30, step=5
        )
        st.caption(f"💡 {t('seconds_per_stop', lang)} ({t('no_stops', lang)} = 0)")

        calculate_button = st.button(f"🏃 {t('calculate', lang)}", type="primary", width='stretch')

    # Main content area
    if calculate_button or 'pacing_data' in st.session_state:
        try:
            # Parse inputs
            if input_mode == t('target_finish_time', lang):
                target_time_min = parse_time_input(finish_time_str)
            else:
                pace_min_km = parse_pace_input(avg_pace_str)
                target_time_min = pace_min_km * total_distance_km

            # Calculate pacing
            pacing_data = calculate_pacing(
                trackpoints, target_time_min, REST_STOPS,
                total_distance_km, gap_adjusted_distance_m,
                power_fade=power_fade, rest_duration_sec=rest_duration
            )
            st.session_state['pacing_data'] = pacing_data
            st.session_state['target_time_min'] = target_time_min

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your input format (HH:MM:SS or MM:SS)")
            return

    # Display results if available
    if 'pacing_data' in st.session_state:
        pacing_data = st.session_state['pacing_data']
        target_time_min = st.session_state.get('target_time_min', 105)

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            f"{t('summary', lang)}",
            f"{t('course_sections', lang)}",
            f"{t('tables', lang)}",
            f"{t('print_export', lang)}"
        ])

        # ==================== TAB 1: SUMMARY ====================
        with tab1:
            # Strategy summary
            st.markdown(f"### {t('target', lang)}: {format_time(target_time_min)} finish time")

            # Strategy
            fade = pacing_data['power_fade']
            if fade != 0:
                fade_dir = t('strategy_positive', lang) if fade > 0 else t('strategy_negative', lang)
                fade_pct = abs(fade) * 0.5
                fade_desc = t('faster_start', lang) if fade > 0 else t('slower_start', lang)
                fade_desc2 = t('slower_finish', lang) if fade > 0 else t('faster_finish', lang)
                st.markdown(f"📍 **{t('strategy', lang)}:** {fade_dir} ({fade:+d})")
                st.markdown(f"   {t('pace_adjustment', lang)}: {fade_pct:.1f}% {fade_desc}, {fade_desc2}")
            else:
                st.markdown(f"📍 **{t('strategy', lang)}:** {t('strategy_even', lang)}")

            # Times
            total_rest_time = pacing_data['total_rest_time_min']
            running_time = pacing_data['running_time_min']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('running_time', lang), format_time(running_time))
            with col2:
                if total_rest_time > 0:
                    st.metric(t('rest_time', lang), f"{format_time(total_rest_time)} ({len(REST_STOPS)} {t('stops', lang)})")
                else:
                    st.metric(t('rest_time', lang), "0:00")
            with col3:
                st.metric(t('total_time', lang), format_time(target_time_min))

            col4, col5 = st.columns(2)
            with col4:
                st.metric(t('avg_pace', lang), f"{format_time(running_time / total_distance_km)}/km")
            with col5:
                st.metric(t('gap_pace', lang), f"{format_time(pacing_data['target_gap_pace'])}/km")

            # Split analysis
            st.markdown(f"### 📊 {t('split_analysis', lang)}")

            halfway_km = total_distance_km / 2
            first_half_time = 0.0
            for split in pacing_data['km_splits']:
                if split['km'] <= halfway_km:
                    first_half_time += split['segment_time_min']
                elif split['km'] > halfway_km:
                    km_start = int(split['km'] - 1) if split['km'] == int(split['km']) else int(split['km'])
                    if km_start < halfway_km:
                        fraction_in_first = halfway_km - km_start
                        first_half_time += split['actual_pace_min_km'] * fraction_in_first
                    break

            second_half_time = pacing_data['calculated_finish_time_min'] - first_half_time
            first_half_distance = halfway_km
            second_half_distance = total_distance_km - halfway_km
            first_half_pace = first_half_time / first_half_distance if first_half_distance > 0 else 0
            second_half_pace = second_half_time / second_half_distance if second_half_distance > 0 else 0

            elev_gain_1st, elev_loss_1st = calculate_elevation_changes(trackpoints, 0, halfway_km)
            elev_gain_2nd, elev_loss_2nd = calculate_elevation_changes(trackpoints, halfway_km, total_distance_km)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"**{t('first_half', lang)} ({first_half_distance:.1f}km)**")
                st.markdown(f"⏱️ {format_time(first_half_time)} @ {format_time(first_half_pace)}/km")
                st.markdown(f"⛰️ +{elev_gain_1st:.0f}m / -{elev_loss_1st:.0f}m")

            with col_right:
                st.markdown(f"**{t('second_half', lang)} ({second_half_distance:.1f}km)**")
                st.markdown(f"⏱️ {format_time(second_half_time)} @ {format_time(second_half_pace)}/km")
                st.markdown(f"⛰️ +{elev_gain_2nd:.0f}m / -{elev_loss_2nd:.0f}m")

            split_diff = second_half_time - first_half_time
            if split_diff > 0:
                st.info(f"**{t('split', lang)}:** +{format_time(abs(split_diff))} ({t('slower_2nd', lang)})")
            elif split_diff < 0:
                st.success(f"**{t('split', lang)}:** -{format_time(abs(split_diff))} ({t('faster_2nd', lang)})")
            else:
                st.info(f"**{t('split', lang)}:** {t('even', lang)}")

            st.divider()

            # Elevation profile
            st.markdown(f"### {t('course_profile', lang)}")
            fig_elevation = plot_elevation_profile(trackpoints, REST_STOPS, total_distance_km, lang)
            st.pyplot(fig_elevation)
            plt.close(fig_elevation)

            # Pace comparison
            st.markdown(f"### {t('pace_per_km', lang)}")
            fig_pace = plot_pace_comparison(pacing_data, total_distance_km, lang)
            st.pyplot(fig_pace)
            plt.close(fig_pace)

        # ==================== TAB 2: COURSE SECTIONS ====================
        with tab2:
            st.markdown(f"### {t('course_sections', lang)}")

            for i, section in enumerate(COURSE_SECTIONS):
                start = section['start_km']
                end = min(section['end_km'], total_distance_km)
                name = section[f'name_{lang}']
                strategy = section[f'strategy_{lang}']
                pacing_tip = section[f'pacing_{lang}']
                icon = section['icon']

                # Calculate section time from rest stops
                if i < len(pacing_data['rest_stops']):
                    section_time = pacing_data['rest_stops'][i]['split_from_prev_min']
                    section_pace = pacing_data['rest_stops'][i]['actual_pace_min_km']
                else:
                    # Last section
                    section_time = pacing_data['calculated_finish_time_min'] - pacing_data['rest_stops'][-1]['arrival_time_min']
                    distance = end - start
                    section_pace = section_time / distance if distance > 0 else 0

                elev_gain, elev_loss = calculate_elevation_changes(trackpoints, start, end)

                with st.expander(f"{icon} {name} ({start:.1f} - {end:.1f} km)"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric(t('time', lang), format_time(section_time))
                    with col_b:
                        st.metric(t('pace', lang), f"{format_time(section_pace)}/km")
                    with col_c:
                        st.markdown(f"**{t('elev', lang)}:** +{elev_gain:.0f}m / -{elev_loss:.0f}m")

                    st.markdown(f"**{t('strategy', lang)}:** {strategy}")
                    st.info(f"💡 **{t('pacing_tip', lang)}:** {pacing_tip}")

        # ==================== TAB 3: TABLES ====================
        with tab3:
            # Rest stops table
            st.markdown(f"### {t('rest_stop_arrival', lang)}")

            rest_data = []
            for stop in pacing_data['rest_stops']:
                rest_data.append({
                    t('rest', lang): stop['stop_number'],
                    'km': f"{stop['distance_km']:.1f}",
                    t('elapsed', lang): stop['elapsed_time_str'],
                    t('split_time', lang): format_time(stop['split_from_prev_min']),
                    t('actual', lang): f"{format_time(stop.get('actual_pace_min_km', 0))}/km",
                    'GAP': f"{format_time(stop.get('gap_pace_min_km', 0))}/km",
                    t('elev_change', lang): f"+{stop.get('elev_gain_m', 0):.0f}/-{stop.get('elev_loss_m', 0):.0f}",
                    t('rest', lang): f"{int(stop['suggested_rest_min'] * 60)}s"
                })

            # Add finish row
            finish_split_time = pacing_data['calculated_finish_time_min'] - pacing_data['rest_stops'][-1]['arrival_time_min']
            finish_distance = total_distance_km - pacing_data['rest_stops'][-1]['distance_km']
            elev_gain_finish, elev_loss_finish = calculate_elevation_changes(
                trackpoints, pacing_data['rest_stops'][-1]['distance_km'], total_distance_km
            )

            rest_data.append({
                t('rest', lang): f"🏁 {t('finish', lang)}",
                'km': f"{total_distance_km:.2f}",
                t('elapsed', lang): format_time(pacing_data['calculated_finish_time_min']),
                t('split_time', lang): format_time(finish_split_time),
                t('actual', lang): f"{format_time(finish_split_time / finish_distance if finish_distance > 0 else 0)}/km",
                'GAP': '-',
                t('elev_change', lang): f"+{elev_gain_finish:.0f}/-{elev_loss_finish:.0f}",
                t('rest', lang): '-'
            })

            df_rest = pd.DataFrame(rest_data)
            st.dataframe(df_rest, width='stretch', hide_index=True)

            st.divider()

            # Kilometer splits table
            st.markdown(f"### {t('km_splits', lang)}")

            splits_data = []
            for split in pacing_data['km_splits']:
                km_start = int(split['km'] - 1) if split['km'] == int(split['km']) else int(split['km'])
                km_end = split['km']
                elev_gain, elev_loss = calculate_elevation_changes(trackpoints, km_start, km_end)

                splits_data.append({
                    'KM': f"{split['km']:.1f}",
                    t('actual', lang): f"{format_time(split['actual_pace_min_km'])}/km",
                    'GAP': f"{format_time(split['gap_pace_min_km'])}/km",
                    t('grade', lang): f"{split['grade_percent']:+.1f}%",
                    t('elev_change', lang): f"+{elev_gain:.0f}/-{elev_loss:.0f}",
                    t('total', lang): format_time(split['cumulative_time_min'])
                })

            df_splits = pd.DataFrame(splits_data)
            st.dataframe(df_splits, width='stretch', hide_index=True)

        # ==================== TAB 4: PRINT/EXPORT ====================
        with tab4:
            st.markdown(f"### {t('quick_ref', lang)}")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(f"#### {t('print_pocket', lang)}")
                pocket_card = generate_pocket_card(pacing_data, total_distance_km)
                st.code(pocket_card, language=None)
                st.download_button(
                    f"📥 {t('download_pocket', lang)}",
                    pocket_card,
                    file_name="race_pacing_pocket_card.txt",
                    mime="text/plain"
                )

            with col_right:
                st.markdown(f"#### {t('print_wrist', lang)}")
                wrist_band = generate_wrist_band(pacing_data, total_distance_km)
                st.code(wrist_band, language=None)
                st.download_button(
                    f"📥 {t('download_wrist', lang)}",
                    wrist_band,
                    file_name="race_pacing_wrist_band.txt",
                    mime="text/plain"
                )

            st.divider()

            # Reference table
            st.markdown("### 📋 Finish Time Reference")

            ref_data = [
                {"Finish Time": "1:30:00", "Avg Pace": "4:16/km", f"{t('level', lang)}": t('elite', lang)},
                {"Finish Time": "1:40:00", "Avg Pace": "4:44/km", f"{t('level', lang)}": t('advanced', lang)},
                {"Finish Time": "1:45:00", "Avg Pace": "5:00/km", f"{t('level', lang)}": t('competitive', lang)},
                {"Finish Time": "1:50:00", "Avg Pace": "5:14/km", f"{t('level', lang)}": t('strong', lang)},
                {"Finish Time": "2:00:00", "Avg Pace": "5:41/km", f"{t('level', lang)}": t('intermediate', lang)},
                {"Finish Time": "2:15:00", "Avg Pace": "6:24/km", f"{t('level', lang)}": t('recreational', lang)},
            ]

            df_ref = pd.DataFrame(ref_data)
            st.dataframe(df_ref, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
