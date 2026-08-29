"""
Unit tests for RailRouteAgent schemas and sample dataset loading.
"""

import json
import pytest
from pydantic import ValidationError

from src.schema import Station, TrainSchedule, Leg, SplitItinerary
from src.data.load_sample_data import generate_sample_dataset, save_dataset
from src.config import TRAIN_NETWORK_JSON


def test_station_schema_valid():
    st = Station(code="NDLS", name="New Delhi", zone="NR", platforms=16)
    assert st.code == "NDLS"
    assert st.platforms == 16


def test_station_schema_invalid():
    with pytest.raises(ValidationError):
        Station(code="NDLS", name="New Delhi", zone="NR", platforms=0)


def test_train_schedule_schema_valid():
    ts = TrainSchedule(
        train_no="12952",
        train_name="New Delhi Rajdhani",
        src_station="NDLS",
        dest_station="CSMT",
        departure_time="16:55",
        arrival_time="08:35",
        day_offset=1,
        classes=["1A", "2A", "3A"],
        avg_delay_mins=15
    )
    assert ts.train_no == "12952"
    assert ts.departure_time == "16:55"
    assert ts.arrival_time == "08:35"


def test_train_schedule_invalid_time():
    with pytest.raises(ValidationError):
        TrainSchedule(
            train_no="12952",
            train_name="Test Train",
            src_station="NDLS",
            dest_station="CSMT",
            departure_time="25:00",  # Invalid hour
            arrival_time="08:35",
            day_offset=0,
            classes=["SL"],
            avg_delay_mins=0
        )


def test_leg_and_split_itinerary_schema():
    ts = TrainSchedule(
        train_no="12002",
        train_name="Bhopal Shatabdi",
        src_station="NDLS",
        dest_station="BPL",
        departure_time="06:00",
        arrival_time="14:05",
        day_offset=0,
        classes=["EC", "CC"],
        avg_delay_mins=10
    )
    leg = Leg(
        train=ts,
        from_station="NDLS",
        to_station="BPL",
        dep_time="06:00",
        arr_time="14:05",
        class_type="CC",
        availability_status="AVAILABLE-0042",
        confirmation_prob=0.98
    )
    assert leg.confirmation_prob == 0.98

    itinerary = SplitItinerary(
        route_id="R-NDLS-BPL-001",
        origin="NDLS",
        destination="BPL",
        legs=[leg],
        total_duration_mins=485,
        layover_buffer_mins=0,
        is_operationally_feasible=True,
        feasibility_notes="Direct flight/train connection feasible.",
        overall_confirmation_prob=0.98
    )
    assert itinerary.origin == "NDLS"
    assert len(itinerary.legs) == 1


def test_dataset_generation_and_serialization(tmp_path):
    dataset = generate_sample_dataset()
    assert dataset["metadata"]["station_count"] >= 10
    assert dataset["metadata"]["train_route_count"] >= 25
    assert dataset["metadata"]["is_connected"] is True

    test_file = tmp_path / "test_network.json"
    save_dataset(test_file)
    assert test_file.exists()

    with open(test_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["station_count"] == dataset["metadata"]["station_count"]
