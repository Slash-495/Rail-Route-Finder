"""
Routing Tools for RailRouteAgent.
Provides optimized 2-hop heuristic search functions to find direct trains and candidate split junctions.
"""

import ast
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import networkx as nx
import pandas as pd

from src.config import TRAIN_NETWORK_JSON, TRAIN_NETWORK_CSV


def calculate_duration_mins(dep_time: str, arr_time: str) -> int:
    """Calculates journey duration in minutes, handling midnight wrap-around via modulo math."""
    fmt = "%H:%M"
    tdelta = datetime.strptime(arr_time, fmt) - datetime.strptime(dep_time, fmt)
    return int((tdelta.total_seconds() % 86400) // 60)


@lru_cache(maxsize=None)
def _load_trains_dataframe(csv_path: Path = TRAIN_NETWORK_CSV, json_path: Path = TRAIN_NETWORK_JSON) -> pd.DataFrame:
    """Loads the actual railway timetable dataset into a Pandas DataFrame from CSV or JSON.

    If dataset file is missing on disk (e.g. fresh ephemeral environment on Streamlit Cloud),
    dynamically generates and persists the dataset on-the-fly.
    """
    csv_file = Path(csv_path) if csv_path else None
    json_file = Path(json_path) if json_path else None

    if csv_file and csv_file.exists():
        df = pd.read_csv(csv_file)
    elif json_file and json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("trains", []))
    else:
        try:
            from src.data.load_sample_data import save_dataset
            data = save_dataset(json_file or TRAIN_NETWORK_JSON)
            df = pd.DataFrame(data.get("trains", []))
        except Exception:
            from src.data.load_sample_data import generate_sample_dataset
            data = generate_sample_dataset()
            df = pd.DataFrame(data.get("trains", []))

    df["src_station"] = df["src_station"].astype(str).str.strip().str.upper()
    df["dest_station"] = df["dest_station"].astype(str).str.strip().str.upper()
    df["train_no"] = df["train_no"].astype(str).str.strip()

    if "classes" in df.columns:
        def _parse_classes(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
                return [c.strip(" '\"[]") for c in val.split(",") if c.strip(" '\"[]")]
            return ["SL", "3A", "2A"]
        df["classes"] = df["classes"].apply(_parse_classes)

    return df


@lru_cache(maxsize=None)
def _load_railway_graph() -> nx.MultiDiGraph:
    """Builds and caches NetworkX MultiDiGraph from the actual railway timetable dataset."""
    df = _load_trains_dataframe()
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        src = str(row["src_station"])
        dest = str(row["dest_station"])
        G.add_edge(src, dest, key=str(row["train_no"]), **row.to_dict())
    return G


@lru_cache(maxsize=None)
def _load_network_data(dataset_path: Path = TRAIN_NETWORK_JSON) -> Dict[str, Any]:
    """Helper to return train network dictionary for backward compatibility."""
    df = _load_trains_dataframe(json_path=dataset_path)
    return {
        "trains": df.to_dict(orient="records")
    }


def reload_network_cache():
    """Clears cached timetable, DataFrame, and Graph indexes to force reload from disk."""
    _load_trains_dataframe.cache_clear()
    _load_railway_graph.cache_clear()
    _load_network_data.cache_clear()


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
    """Finds all direct trains operating between source and destination stations using dynamic Pandas filtering.

    Args:
        src: Origin station code.
        dest: Destination station code.
        date: Travel date string in YYYY-MM-DD format (optional).

    Returns:
        List[Dict[str, Any]]: List of direct train schedule dictionaries matching origin and destination.
    """
    df = _load_trains_dataframe()
    src_clean = str(src).strip().upper()
    dest_clean = str(dest).strip().upper()

    direct_df = df[(df["src_station"] == src_clean) & (df["dest_station"] == dest_clean)]
    trains = []
    for _, row in direct_df.iterrows():
        trains.append({
            "train_no": str(row["train_no"]),
            "train_name": str(row["train_name"]),
            "src_station": str(row["src_station"]),
            "dest_station": str(row["dest_station"]),
            "departure_time": str(row["departure_time"]),
            "arrival_time": str(row["arrival_time"]),
            "day_offset": int(row.get("day_offset", 0)),
            "classes": row["classes"],
            "avg_delay_mins": int(row.get("avg_delay_mins", 0)),
            "availability_status": str(row.get("availability_status", "AVAILABLE-0010")),
            "confirmation_prob": float(row.get("confirmation_prob", 0.90)),
        })
    return trains


def find_split_junctions(
    src: str,
    dest: str,
    max_layover_hrs: int = 6,
    min_layover_mins: int = 30
) -> List[Dict[str, Any]]:
    """Finds 2-leg split journey connections (A -> B -> C) between source and destination.

    Loads the actual railway dataset (CSV/Graph) and executes dynamic Pandas/Graph filtering
    to find valid intermediate transfer stations satisfying layover buffer constraints.
    Zero hardcoded station names or specific routes are used.

    Args:
        src: Origin station code.
        dest: Destination station code.
        max_layover_hrs: Maximum allowable scheduled transfer buffer in hours (default 6).
        min_layover_mins: Minimum required scheduled transfer buffer in minutes (default 30).

    Returns:
        List[Dict[str, Any]]: List of candidate split itinerary dictionaries containing junction code,
        leg 1 train details, leg 2 train details, scheduled layover in minutes, and total duration.
    """
    df = _load_trains_dataframe()
    G = _load_railway_graph()

    src_clean = str(src).strip().upper()
    dest_clean = str(dest).strip().upper()

    candidates: List[Dict[str, Any]] = []

    if df.empty or src_clean == dest_clean:
        print(f"Found {len(candidates)} dynamic candidates")
        return candidates

    # 1. Graph-based dynamic intermediate transfer station discovery
    reachable_from_src = set(G.successors(src_clean)) if G.has_node(src_clean) else set()
    connects_to_dest = set(G.predecessors(dest_clean)) if G.has_node(dest_clean) else set()
    valid_intermediate_stations = reachable_from_src.intersection(connects_to_dest)
    valid_intermediate_stations.discard(src_clean)
    valid_intermediate_stations.discard(dest_clean)

    # 2. Dynamic Pandas filtering: filter leg 1 and leg 2 DataFrames
    leg1_df = df[df["src_station"] == src_clean].copy()
    leg2_df = df[df["dest_station"] == dest_clean].copy()

    if leg1_df.empty or leg2_df.empty:
        print(f"Found {len(candidates)} dynamic candidates")
        return candidates

    # Dynamic join on intermediate transfer station
    merged_df = pd.merge(
        leg1_df,
        leg2_df,
        left_on="dest_station",
        right_on="src_station",
        suffixes=("_leg1", "_leg2")
    )

    if merged_df.empty:
        print(f"Found {len(candidates)} dynamic candidates")
        return candidates

    # Strictly filter out direct trips and cyclic routes
    merged_df = merged_df[
        (merged_df["dest_station_leg1"] != src_clean) &
        (merged_df["dest_station_leg1"] != dest_clean)
    ].copy()

    if valid_intermediate_stations:
        merged_df = merged_df[merged_df["dest_station_leg1"].isin(valid_intermediate_stations)].copy()

    if merged_df.empty:
        print(f"Found {len(candidates)} dynamic candidates")
        return candidates

    # 3. Dynamic Layover & Journey Duration Calculations
    def _to_mins(t_str: str) -> int:
        h, m = map(int, str(t_str).split(":"))
        return h * 60 + m

    arr1_mins = merged_df["arrival_time_leg1"].apply(_to_mins)
    dep2_mins = merged_df["departure_time_leg2"].apply(_to_mins)

    # Calculate layover handling midnight wrap-around via modulo 1440
    layovers = (dep2_mins - arr1_mins) % 1440
    merged_df["scheduled_layover_mins"] = layovers

    # Durations of leg 1 and leg 2
    dep1_mins = merged_df["departure_time_leg1"].apply(_to_mins)
    arr2_mins = merged_df["arrival_time_leg2"].apply(_to_mins)
    dur1 = (arr1_mins - dep1_mins) % 1440
    dur2 = (arr2_mins - dep2_mins) % 1440
    merged_df["total_duration_mins"] = dur1 + merged_df["scheduled_layover_mins"] + dur2

    # 4. Dynamic Filtering on Layover Window Constraint
    max_layover_mins = max_layover_hrs * 60

    # Pass 1: Standard layover window [min_layover_mins, max_layover_mins]
    filtered_df = merged_df[
        (merged_df["scheduled_layover_mins"] >= min_layover_mins) &
        (merged_df["scheduled_layover_mins"] <= max_layover_mins)
    ].copy()

    # Pass 2: Fallback relaxation if 0 candidates initially found
    if filtered_df.empty:
        relaxed_max = max(max_layover_mins * 2, 720)
        filtered_df = merged_df[
            (merged_df["scheduled_layover_mins"] >= 15) &
            (merged_df["scheduled_layover_mins"] <= relaxed_max)
        ].copy()

    # Pass 3: Wide day-window fallback
    if filtered_df.empty:
        filtered_df = merged_df[
            (merged_df["scheduled_layover_mins"] > 0) &
            (merged_df["scheduled_layover_mins"] <= 1440)
        ].copy()

    # Sort candidates descending by layover safety buffer and ascending by total travel duration
    filtered_df = filtered_df.sort_values(
        by=["scheduled_layover_mins", "total_duration_mins"],
        ascending=[False, True]
    )

    # 5. Build candidate dictionaries
    for _, row in filtered_df.iterrows():
        t1 = {
            "train_no": str(row["train_no_leg1"]),
            "train_name": str(row["train_name_leg1"]),
            "src_station": str(row["src_station_leg1"]),
            "dest_station": str(row["dest_station_leg1"]),
            "departure_time": str(row["departure_time_leg1"]),
            "arrival_time": str(row["arrival_time_leg1"]),
            "day_offset": int(row.get("day_offset_leg1", 0)),
            "classes": row["classes_leg1"],
            "avg_delay_mins": int(row.get("avg_delay_mins_leg1", 0)),
            "availability_status": str(row.get("availability_status_leg1", "AVAILABLE-0010")),
            "confirmation_prob": float(row.get("confirmation_prob_leg1", 0.90)),
        }
        t2 = {
            "train_no": str(row["train_no_leg2"]),
            "train_name": str(row["train_name_leg2"]),
            "src_station": str(row["src_station_leg2"]),
            "dest_station": str(row["dest_station_leg2"]),
            "departure_time": str(row["departure_time_leg2"]),
            "arrival_time": str(row["arrival_time_leg2"]),
            "day_offset": int(row.get("day_offset_leg2", 0)),
            "classes": row["classes_leg2"],
            "avg_delay_mins": int(row.get("avg_delay_mins_leg2", 0)),
            "availability_status": str(row.get("availability_status_leg2", "AVAILABLE-0010")),
            "confirmation_prob": float(row.get("confirmation_prob_leg2", 0.90)),
        }
        junc = str(row["dest_station_leg1"]).strip().upper()

        candidates.append({
            "junction": junc,
            "train_1": t1,
            "train_2": t2,
            "scheduled_layover_mins": int(row["scheduled_layover_mins"]),
            "total_duration_mins": int(row["total_duration_mins"]),
        })

    print(f"Found {len(candidates)} dynamic candidates")
    return candidates
