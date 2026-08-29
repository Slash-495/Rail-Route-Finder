"""
Unit tests for ZeroShotBaselineAgent (Iteration 0).
"""

import json
import pytest
from src.agents.baseline_agent import ZeroShotBaselineAgent
from src.config import TRAIN_NETWORK_JSON


def test_baseline_agent_prompt_building():
    agent = ZeroShotBaselineAgent()
    prompt = agent.build_prompt("NDLS", "CSMT", "3A", 1)

    assert "NDLS" in prompt
    assert "CSMT" in prompt
    assert "TIMETABLE_DATABASE" in prompt
    assert "SplitItinerary" in prompt or "route_id" in prompt


def test_baseline_agent_solve_and_parse():
    agent = ZeroShotBaselineAgent()
    itin, prompt, raw_response = agent.solve("NDLS", "CSMT", "3A", 1)

    assert prompt != ""
    assert raw_response != ""
    assert itin is not None
    assert itin.origin == "NDLS"
    assert itin.destination == "CSMT"
    assert len(itin.legs) == 2


def test_parse_llm_json_invalid():
    agent = ZeroShotBaselineAgent()
    parsed = agent._parse_llm_json("Invalid JSON content from LLM")
    assert parsed is None
