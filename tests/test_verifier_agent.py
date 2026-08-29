"""
Unit tests for RailRouteAgent VerifierAgent and reflection loop.
"""

import pytest
from src.agents import VerifierAgent, PlannerAgent, VerificationResult


def test_verifier_agent_initialization():
    verifier = VerifierAgent()
    assert verifier.model_name == "gemini-2.5-flash"
    assert verifier.get_trajectory() == []
    assert verifier.get_refinement_history() == []


def test_verifier_agent_verify_and_refine_standard():
    verifier = VerifierAgent()
    results = verifier.verify_and_refine(origin="NDLS", destination="MAO", date="2026-09-15")

    assert len(results) >= 1
    first = results[0]
    assert first["origin"] == "NDLS"
    assert first["destination"] == "MAO"
    assert first["is_operationally_feasible"] is True
    assert "VERIFIED" in first["feasibility_notes"]


def test_verifier_agent_dialogue_history_logging():
    verifier = VerifierAgent()
    verifier.verify_and_refine(origin="NDLS", destination="CSMT")

    history = verifier.get_refinement_history()
    assert len(history) >= 1

    first_turn = history[0]
    assert "iteration" in first_turn
    assert "proposed_routes" in first_turn
    assert "audit_results" in first_turn
    assert "critiques" in first_turn


def test_verifier_agent_rejects_tight_trap():
    # Construct PlannerAgent with tight trap scenario
    verifier = VerifierAgent()
    results = verifier.verify_and_refine(origin="NDLS", destination="CSMT", date="2026-09-15")

    # Ensure all returned itineraries are operationally verified
    for res in results:
        assert res["is_operationally_feasible"] is True
        assert res["layover_buffer_mins"] >= 45
