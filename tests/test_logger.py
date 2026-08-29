"""
Unit tests for RailRouteAgent TrajectoryLogger telemetry engine.
"""

import json
import pytest
from src.utils.logger import TrajectoryLogger, get_logger


def test_logger_singleton():
    logger1 = TrajectoryLogger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_logger_event_recording():
    logger = get_logger()
    logger.clear()
    assert len(logger.events) == 0

    logger.log_event("Planner", "prompt", {"query": "NDLS -> MAO"})
    logger.log_event("Verifier", "tool_call", {"tool": "calculate_connection_risk"})
    logger.log_event("System", "human_checkpoint", {"action": "SIMULATE_BOOKING", "choice": "1"})

    assert len(logger.events) == 3
    assert logger.events[0]["agent"] == "Planner"
    assert logger.events[0]["event_type"] == "prompt"
    assert logger.events[1]["agent"] == "Verifier"
    assert logger.events[2]["agent"] == "System"
    assert logger.events[2]["event_type"] == "human_checkpoint"


def test_logger_export_trajectory(tmp_path):
    logger = get_logger()
    logger.clear()
    logger.log_event("Planner", "prompt", {"test": "data"})

    exported_path = logger.export_trajectory("TEST_SESSION_123")
    assert exported_path.exists()
    assert "run_TEST_SESSION_123.json" in str(exported_path)

    with open(exported_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["agent"] == "Planner"
