"""
Operational Risk & Delay Tools for RailRouteAgent.
Provides functions to calculate dynamic junction transfer buffers, lookup NTES historical delays,
and evaluate operational connection feasibility risk.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import TRAIN_NETWORK_JSON

# Mandatory junction minimum transfer buffer thresholds (in minutes) based on size & complexity
MAJOR_JUNCTION_BUFFERS: Dict[str, int] = {
    "NDLS": 60,  # New Delhi (16 platforms, heavy congestion)
    "CSMT": 60,  # Mumbai CSMT (18 platforms)
    "HWH": 60,   # Howrah Junction (23 platforms)
    "BPL": 45,   # Bhopal Junction (6 platforms)
    "NGP": 45,   # Nagpur Junction (8 platforms)
    "ET": 45,    # Itarsi Junction (7 platforms)
    "PUNE": 45,  # Pune Junction (6 platforms)
    "JBP": 45,   # Jabalpur Junction (6 platforms)
    "CNB": 45,   # Kanpur Central (10 platforms)
    "PRYJ": 45,  # Prayagraj Junction (10 platforms)
    "BSB": 45,   # Varanasi Junction (9 platforms)
    "ADI": 45,   # Ahmedabad Junction (12 platforms)
}

DEFAULT_JUNCTION_BUFFER_MINS: int = 30


def _load_network_data(dataset_path: Path = TRAIN_NETWORK_JSON) -> Dict[str, Any]:
    """Internal helper to load train network dataset."""
    if not dataset_path.exists():
        return {"stations": [], "trains": []}
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dynamic_junction_buffer(station_code: str) -> int:
    """Returns the minimum safe layover buffer in minutes for a junction station.

    Calculates safe transfer time based on station platform size, layout complexity,
    and zone congestion. Major junctions require 45-60 mins, while smaller stations default to 30 mins.

    Args:
        station_code: Unique 3-4 letter railway station code (e.g. "NDLS", "BPL", "MAO").

    Returns:
        int: Mandatory minimum transfer buffer in minutes.
    """
    code_upper = station_code.upper()
    return MAJOR_JUNCTION_BUFFERS.get(code_upper, DEFAULT_JUNCTION_BUFFER_MINS)


def get_historical_delay(train_no: str, station_code: str = "") -> Dict[str, Any]:
    """Mocks historical NTES delay lookup for a given train number and station stop.

    Calculates P50 (median) and P90 (90th percentile) arrival delays deterministically.
    Applies terminal slack time recovery if the station is the final destination.

    Args:
        train_no: 5-digit train number (e.g. "12952", "12616").
        station_code: Station code where arrival delay is evaluated (optional).

    Returns:
        Dict[str, Any]: Dictionary containing train number, station code, P50 delay in mins,
        P90 delay in mins, and boolean indicating if terminal slack recovery was applied.
    """
    data = _load_network_data()
    trains = data.get("trains", [])

    # Find train in dataset if available
    t_data = next((t for t in trains if t["train_no"] == train_no), None)

    if t_data and "avg_delay_mins" in t_data:
        p50 = t_data["avg_delay_mins"]
        p90 = p50 + 20
    else:
        # Deterministic fallback based on train number hash
        hash_val = abs(hash(train_no)) % 100
        if train_no.startswith(("12", "22")):
            # Superfast / Rajdhani / Shatabdi trains (moderate delays: 10..25 mins)
            p50 = (hash_val % 15) + 10
            p90 = p50 + 15
        else:
            # Standard Express / Mail trains (higher delays: 40..85 mins)
            p50 = (hash_val % 35) + 40
            p90 = p50 + 30

    # Terminal slack time recovery logic
    has_terminal_slack = False
    if t_data and station_code and t_data.get("dest_station") == station_code.upper():
        has_terminal_slack = True
        p50 = max(0, p50 - 10)
        p90 = max(5, p90 - 15)

    return {
        "train_no": train_no,
        "station_code": station_code.upper(),
        "p50_delay_mins": p50,
        "p90_delay_mins": p90,
        "has_terminal_slack": has_terminal_slack
    }


def calculate_connection_risk(train_1: str, train_2: str, junction: str, scheduled_layover_mins: int) -> Dict[str, Any]:
    """Calculates operational connection feasibility and risk score for a split transfer.

    Subtracts the P90 historical delay of Train 1 from the scheduled layover time.
    If the effective buffer is less than the dynamic junction buffer requirement,
    returns an operational infeasibility result with detailed rationale.

    Args:
        train_1: Train number of first leg (incoming train).
        train_2: Train number of second leg (outgoing connecting train).
        junction: Transfer junction station code (e.g. "BPL", "NGP").
        scheduled_layover_mins: Scheduled transfer time between arrival of Leg 1 and departure of Leg 2.

    Returns:
        Dict[str, Any]: Dictionary containing feasibility boolean, risk level ("LOW", "HIGH", "CRITICAL"),
        effective buffer in minutes, required buffer in minutes, and detailed explanation string.
    """
    junc_upper = junction.upper()
    req_buffer = get_dynamic_junction_buffer(junc_upper)
    delay_info = get_historical_delay(train_1, junc_upper)
    p90_delay = delay_info["p90_delay_mins"]

    effective_buffer = scheduled_layover_mins - p90_delay
    is_feasible = effective_buffer >= req_buffer

    if not is_feasible:
        if effective_buffer < 0:
            risk_level = "CRITICAL"
            reason = (f"High risk of missed connection due to historical delay: "
                      f"Train {train_1} has P90 delay of {p90_delay}m. Scheduled layover of {scheduled_layover_mins}m "
                      f"results in a negative effective buffer of {effective_buffer}m at {junc_upper}.")
        else:
            risk_level = "HIGH"
            reason = (f"High risk of missed connection due to historical delay: "
                      f"Effective buffer ({effective_buffer}m) after P90 delay ({p90_delay}m) is below "
                      f"the mandatory {req_buffer}m threshold for junction {junc_upper}.")
    else:
        risk_level = "LOW"
        reason = (f"Connection is operationally feasible at {junc_upper}: "
                  f"Effective buffer of {effective_buffer}m (after {p90_delay}m P90 delay) "
                  f"meets the required {req_buffer}m threshold.")

    return {
        "is_feasible": is_feasible,
        "risk_level": risk_level,
        "train_1": train_1,
        "train_2": train_2,
        "junction": junc_upper,
        "scheduled_layover_mins": scheduled_layover_mins,
        "p90_delay_mins": p90_delay,
        "effective_buffer_mins": effective_buffer,
        "required_buffer_mins": req_buffer,
        "reason": reason
    }
