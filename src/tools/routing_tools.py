"""
Graph Routing Tools for RailRouteAgent.
Provides network graph traversal functions to find direct trains and candidate split junctions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import networkx as nx

from src.config import TRAIN_NETWORK_JSON


def calculate_duration_mins(dep_time: str, arr_time: str) -> int:
    """Calculates journey duration in minutes, handling midnight wrap-around via modulo math."""
    fmt = "%H:%M"
    tdelta = datetime.strptime(arr_time, fmt) - datetime.strptime(dep_time, fmt)
    return int((tdelta.total_seconds() % 86400) // 60)


def _load_network_data(dataset_path: Path = TRAIN_NETWORK_JSON) -> Dict[str, Any]:
    """Internal helper to load train network JSON dataset."""
    if not dataset_path.exists():
        return {"stations": [], "trains": []}
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    """Finds all direct trains operating between source and destination stations.

    Args:
        src: Origin station code (e.g. "NDLS", "CSMT").
        dest: Destination station code (e.g. "BPL", "MAO").
        date: Travel date string in YYYY-MM-DD format (optional).

    Returns:
        List[Dict[str, Any]]: List of direct train schedule dictionaries matching origin and destination.
    """
    data = _load_network_data()
    trains = data.get("trains", [])

    direct_matches = []
    for t in trains:
        if t.get("src_station") == src.upper() and t.get("dest_station") == dest.upper():
            direct_matches.append(t)

    return direct_matches


def find_split_junctions(src: str, dest: str, max_layover_hrs: int = 6) -> List[Dict[str, Any]]:
    """Finds 2-leg split journey connections (A -> B -> C) between source and destination.

    Traverses the train network graph strictly for valid candidate paths where
    the scheduled layover at the intermediate junction station is within limit.

    Args:
        src: Origin station code (e.g. "NDLS").
        dest: Destination station code (e.g. "MAO").
        max_layover_hrs: Maximum allowable scheduled transfer buffer in hours (default 6).

    Returns:
        List[Dict[str, Any]]: List of candidate split itinerary dictionaries containing junction code,
        leg 1 train details, leg 2 train details, scheduled layover in minutes, and total duration.
    """
    data = _load_network_data()
    stations = data.get("stations", [])
    trains = data.get("trains", [])

    src_code = src.upper()
    dest_code = dest.upper()
    max_layover_mins = max_layover_hrs * 60

    # Build NetworkX MultiDiGraph
    G = nx.MultiDiGraph()
    for s in stations:
        G.add_node(s["code"], **s)

    for tr in trains:
        G.add_edge(tr["src_station"], tr["dest_station"], key=tr["train_no"], **tr)

    candidate_routes: List[Dict[str, Any]] = []

    # Find first-leg candidates originating at src
    first_legs = [t for t in trains if t["src_station"] == src_code]
    second_legs = [t for t in trains if t["dest_station"] == dest_code]

    for t1 in first_legs:
        junc = t1["dest_station"]
        if junc == dest_code:
            continue  # Skip direct routes

        # Matching second legs from junction to destination
        matching_t2 = [t for t in second_legs if t["src_station"] == junc]

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

    return candidate_routes
