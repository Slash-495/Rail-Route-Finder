"""
Unit tests for RailRouteAgent graph routing and operational risk tools.
"""

import pytest
from src.tools import (
    find_direct_trains,
    find_split_junctions,
    get_dynamic_junction_buffer,
    get_historical_delay,
    calculate_connection_risk,
)


def test_find_direct_trains():
    trains = find_direct_trains("NDLS", "CSMT")
    assert len(trains) >= 1
    assert any(t["train_no"] == "12952" for t in trains)


def test_find_split_junctions():
    splits = find_split_junctions("NDLS", "MAO", max_layover_hrs=6)
    assert len(splits) >= 1
    # Ensure all splits are valid 2-leg routes
    for s in splits:
        assert "junction" in s
        assert "train_1" in s
        assert "train_2" in s
        assert s["scheduled_layover_mins"] > 0
        assert s["scheduled_layover_mins"] <= 360


def test_get_dynamic_junction_buffer():
    assert get_dynamic_junction_buffer("NDLS") == 60
    assert get_dynamic_junction_buffer("CSMT") == 60
    assert get_dynamic_junction_buffer("BPL") == 45
    assert get_dynamic_junction_buffer("PUNE") == 45
    assert get_dynamic_junction_buffer("UNKNOWN_STATION") == 30


def test_get_historical_delay():
    delay_12952 = get_historical_delay("12952", "CSMT")
    assert delay_12952["p50_delay_mins"] >= 0
    assert delay_12952["p90_delay_mins"] > delay_12952["p50_delay_mins"]
    assert delay_12952["has_terminal_slack"] is True


def test_calculate_connection_risk_feasible():
    # Large 200 min layover at BPL
    result = calculate_connection_risk(
        train_1="12002",
        train_2="12061",
        junction="BPL",
        scheduled_layover_mins=200
    )
    assert result["is_feasible"] is True
    assert result["risk_level"] == "LOW"
    assert result["required_buffer_mins"] == 45


def test_calculate_connection_risk_infeasible_trap():
    # Tight 10 min layover at BPL (Shatabdi arrival 14:05 -> connecting train 14:15)
    result = calculate_connection_risk(
        train_1="12002",
        train_2="12190",
        junction="BPL",
        scheduled_layover_mins=10
    )
    assert result["is_feasible"] is False
    assert result["reason"].startswith("High risk of missed connection")
    assert result["required_buffer_mins"] == 45
