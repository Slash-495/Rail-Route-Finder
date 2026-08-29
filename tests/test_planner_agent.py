"""
Unit tests for RailRouteAgent PlannerAgent.
"""

import pytest
from src.agents import PlannerAgent, ProposedRouteResponse, CandidateRoute


def test_planner_agent_initialization():
    agent = PlannerAgent()
    assert agent.model_name == "gemini-2.5-flash"
    assert agent.get_trajectory() == []


def test_planner_agent_plan_route():
    agent = PlannerAgent()
    result = agent.plan_route(origin="NDLS", destination="MAO", date="2026-09-15")

    assert result["origin"] == "NDLS"
    assert result["destination"] == "MAO"
    assert "candidate_routes" in result
    assert len(result["candidate_routes"]) <= 3
    assert len(result["candidate_routes"]) > 0

    first_candidate = result["candidate_routes"][0]
    assert "junction" in first_candidate
    assert "train_1_no" in first_candidate
    assert "train_2_no" in first_candidate
    assert first_candidate["scheduled_layover_mins"] > 0


def test_planner_agent_trajectory_logging():
    agent = PlannerAgent()
    agent.plan_route(origin="NDLS", destination="CSMT", date="2026-09-15")

    trajectory = agent.get_trajectory()
    assert len(trajectory) >= 2
    assert any(step.get("step") == "user_input" for step in trajectory)
    assert any(step.get("step") == "tool_call" and step.get("tool") == "find_split_junctions" for step in trajectory)


def test_planner_agent_max_3_candidates():
    agent = PlannerAgent()
    result = agent.plan_route(origin="NDLS", destination="CSMT")
    assert len(result["candidate_routes"]) <= 3


def test_planner_respects_rejected_junctions():
    agent = PlannerAgent()
    # Route NDLS -> MAO normally yields CSMT candidate routes
    normal_result = agent.plan_route(origin="NDLS", destination="MAO", date="2026-09-15")
    assert any(c["junction"] == "CSMT" for c in normal_result["candidate_routes"])

    # Now pass rejected_junctions=["CSMT"]
    filtered_result = agent.plan_route(
        origin="NDLS",
        destination="MAO",
        date="2026-09-15",
        rejected_junctions=["CSMT"]
    )
    # Verify no candidate route uses CSMT
    for candidate in filtered_result["candidate_routes"]:
        assert candidate["junction"] != "CSMT"

    # Verify that the trajectory log records the negative constraint in system prompt
    trajectory = agent.get_trajectory()
    user_input_step = next(s for s in trajectory if s.get("step") == "user_input")
    assert "DO NOT route through these stations: ['CSMT']" in user_input_step["system_prompt"]
