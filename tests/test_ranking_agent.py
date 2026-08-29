"""
Unit tests for RailRouteAgent RankingAgent.
"""

import pytest
from src.agents import RankingAgent, VerifierAgent


def test_ranking_agent_initialization():
    ranker = RankingAgent()
    assert ranker.model_name == "gemini-2.5-flash"
    assert ranker.get_trajectory() == []


def test_ranking_agent_rank_and_summarize():
    verifier = VerifierAgent()
    verified_routes = verifier.verify_and_refine(origin="NDLS", destination="MAO")

    ranker = RankingAgent()
    report = ranker.rank_and_summarize(verified_routes)

    assert "RailRouteAgent" in report
    assert "Option 1" in report
    assert "`NDLS`" in report or "NDLS" in report
    assert "`MAO`" in report or "MAO" in report


def test_ranking_agent_empty_routes():
    ranker = RankingAgent()
    report = ranker.rank_and_summarize([])
    assert "No operationally feasible" in report


def test_ranking_agent_trajectory_logging():
    verifier = VerifierAgent()
    verified_routes = verifier.verify_and_refine(origin="NDLS", destination="CSMT")

    ranker = RankingAgent()
    ranker.rank_and_summarize(verified_routes)

    trajectory = ranker.get_trajectory()
    assert len(trajectory) >= 1
    assert any(step.get("step") in ("input_payload", "llm_output", "fallback_output") for step in trajectory)
