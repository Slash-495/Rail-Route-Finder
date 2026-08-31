"""
Routing Tools for RailRouteAgent.
Provides optimized 2-hop heuristic search functions to find direct trains and candidate split junctions.
"""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.config import TRAIN_NETWORK_JSON


def calculate_duration_mins(dep_time: str, arr_time: str) -> int:
    """Calculates journey duration in minutes, handling midnight wrap-around via modulo math."""
    fmt = "%H:%M"
    tdelta = datetime.strptime(arr_time, fmt) - datetime.strptime(dep_time, fmt)
    return int((tdelta.total_seconds() % 86400) // 60)


@lru_cache(maxsize=None)
def _load_network_data(dataset_path: Path = TRAIN_NETWORK_JSON) -> Dict[str, Any]:
    """Internal helper to load train network JSON dataset cached in memory via LRU cache."""
    if not dataset_path.exists():
        return {"stations": [], "trains": []}
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _get_train_indexes() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Builds and caches station index dictionaries for O(1) lookups: (by_src, by_dest)."""
    data = _load_network_data()
    trains = data.get("trains", [])

    by_src: Dict[str, List[Dict[str, Any]]] = {}
    by_dest: Dict[str, List[Dict[str, Any]]] = {}

    for t in trains:
        src = t.get("src_station", "").upper()
        dest = t.get("dest_station", "").upper()
        by_src.setdefault(src, []).append(t)
        by_dest.setdefault(dest, []).append(t)

    return by_src, by_dest


def parse_time_to_minutes(time_str: str) -> int:
    """Helper function to convert HH:MM time string into minutes past midnight.

    Args:
        time_str: Time string in HH:MM format.

    Returns:
        int: Minutes since 00:00 past midnight.
    """
    hh, mm = map(int, time_str.split(":"))
    return hh * 60 + mm


def find_direct_trains(src: str, dest: str, date: str = "") -> List[Dict[str, Any]]:
    """Finds all direct trains operating between source and destination stations using O(1) index lookup.

    Args:
        src: Origin station code (e.g. "NDLS", "CSMT").
        dest: Destination station code (e.g. "BPL", "MAO").
        date: Travel date string in YYYY-MM-DD format (optional).

    Returns:
        List[Dict[str, Any]]: List of direct train schedule dictionaries matching origin and destination.
    """
    by_src, _ = _get_train_indexes()
    src_code = src.upper()
    dest_code = dest.upper()

    return [t for t in by_src.get(src_code, []) if t.get("dest_station", "").upper() == dest_code]


def find_split_junctions(src: str, dest: str, max_layover_hrs: int = 6) -> List[Dict[str, Any]]:
    """Finds 2-leg split journey connections (A -> B -> C) between source and destination.

    Uses an optimized 2-hop heuristic search indexed by station codes. Results are sorted
    descending by layover safety buffer and ascending by total duration before returning.

    Args:
        src: Origin station code (e.g. "NDLS").
        dest: Destination station code (e.g. "MAO").
        max_layover_hrs: Maximum allowable scheduled transfer buffer in hours (default 6).

    Returns:
        List[Dict[str, Any]]: List of candidate split itinerary dictionaries containing junction code,
        leg 1 train details, leg 2 train details, scheduled layover in minutes, and total duration.
    """
    by_src, by_dest = _get_train_indexes()
    src_code = src.upper()
    dest_code = dest.upper()
    max_layover_mins = max_layover_hrs * 60

    candidate_routes: List[Dict[str, Any]] = []

    first_legs = by_src.get(src_code, [])
    second_legs_dest = by_dest.get(dest_code, [])

    # Group second leg candidates by junction source station for O(1) matching
    second_legs_by_junc: Dict[str, List[Dict[str, Any]]] = {}
    for t2 in second_legs_dest:
        junc_src = t2.get("src_station", "").upper()
        second_legs_by_junc.setdefault(junc_src, []).append(t2)

    for t1 in first_legs:
        junc = t1.get("dest_station", "").upper()
        if junc == dest_code or junc == src_code:
            continue  # Skip direct or cyclic routes

        matching_t2 = second_legs_by_junc.get(junc, [])

        for t2 in matching_t2:
            arr1 = parse_time_to_minutes(t1["arrival_time"])
            dep2 = parse_time_to_minutes(t2["departure_time"])

            if dep2 > arr1:
                layover = dep2 - arr1
            else:
                layover = (dep2 + 1440) - arr1

            if 0 < layover <= max_layover_mins:
                dur1 = calculate_duration_mins(t1["departure_time"], t1["arrival_time"])
                dur2 = calculate_duration_mins(t2["departure_time"], t2["arrival_time"])
                total_duration = dur1 + layover + dur2

                candidate_routes.append({
                    "junction": junc,
                    "train_1": t1,
                    "train_2": t2,
                    "scheduled_layover_mins": layover,
                    "total_duration_mins": total_duration
                })

    # Sort candidates descending by layover safety buffer and ascending by total travel duration
    candidate_routes.sort(key=lambda c: (-c["scheduled_layover_mins"], c["total_duration_mins"]))

    return candidate_routes
