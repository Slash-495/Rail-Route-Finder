"""
Unit tests for RailRouteAgent master orchestrator CLI (src/main.py).
"""

import pytest
from src.main import run_pipeline


def test_run_pipeline_non_interactive():
    status = run_pipeline(origin="NDLS", destination="MAO", date="2026-09-15", interactive=False)
    assert status == 0


def test_run_pipeline_invalid_stations():
    status = run_pipeline(origin="INVALID_SRC", destination="INVALID_DEST", date="2026-09-15", interactive=False)
    assert status == 1
