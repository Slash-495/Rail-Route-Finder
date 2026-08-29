"""
Unit tests for RailRouteAgent benchmark evaluator.
"""

import json
import pytest

from benchmarks.evaluator import BenchmarkEvaluator, TestCase, BaselineSolver, AgentSolver
from src.config import TEST_CASES_JSON, TRAIN_NETWORK_JSON


def test_test_cases_schema_validation():
    evaluator = BenchmarkEvaluator(test_cases_path=TEST_CASES_JSON)
    cases = evaluator.load_and_validate_test_cases()
    assert len(cases) == 12

    # Verify edge cases presence
    tc11 = next(c for c in cases if c.id == "TC-11")
    assert tc11.scenario_type == "Tight Buffer Trap"
    assert tc11.difficulty == "edge_case"

    tc12 = next(c for c in cases if c.id == "TC-12")
    assert tc12.scenario_type == "Cascading Delay Risk"
    assert tc12.difficulty == "edge_case"


def test_baseline_and_agent_evaluator_runs():
    evaluator = BenchmarkEvaluator(test_cases_path=TEST_CASES_JSON, network_path=TRAIN_NETWORK_JSON)
    results = evaluator.run_evaluation()

    assert "zero_shot_llm_baseline" in results["summary_metrics"]
    assert "heuristic_baseline" in results["summary_metrics"]
    assert "agent" in results["summary_metrics"]

    b_llm = results["summary_metrics"]["zero_shot_llm_baseline"]
    h_m = results["summary_metrics"]["heuristic_baseline"]
    a_m = results["summary_metrics"]["agent"]

    assert a_m["viable_route_found_rate"] == 100.0
    assert a_m["operational_feasibility_pass_rate"] == 100.0
    assert b_llm["operational_feasibility_pass_rate"] < 100.0  # Zero-shot LLM fails on edge case traps


def test_evaluator_save_results(tmp_path):
    output_file = tmp_path / "test_benchmark_results.json"
    evaluator = BenchmarkEvaluator(test_cases_path=TEST_CASES_JSON, network_path=TRAIN_NETWORK_JSON)
    results = evaluator.run_evaluation()

    evaluator.save_results(results, output_path=output_file)
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "summary_metrics" in data
    assert data["summary_metrics"]["agent"]["operational_feasibility_pass_rate"] == 100.0
