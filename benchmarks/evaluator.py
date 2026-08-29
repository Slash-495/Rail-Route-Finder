"""
Automated Scoring Engine & Benchmark Evaluator for RailRouteAgent.
Evaluates zero-shot LLM baseline (Iteration 0), heuristic solver, and agentic split-journey planner.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import TEST_CASES_JSON, TRAIN_NETWORK_JSON, LOGS_DIR, TRAJECTORIES_DIR
from src.schema import Station, TrainSchedule, Leg, SplitItinerary
from src.agents.baseline_agent import ZeroShotBaselineAgent


class TestCase(BaseModel):
    """Pydantic model for validating benchmark test cases."""
    __test__ = False
    id: str = Field(..., description="Test case identifier e.g. TC-01")
    origin: str = Field(..., description="Origin station code")
    destination: str = Field(..., description="Destination station code")
    travel_date_offset_days: int = Field(..., ge=1, le=30, description="Date offset in days")
    preferred_class: str = Field(..., description="Preferred class e.g. 3A, 2A, SL")
    direct_train_status: str = Field(..., description="Status of direct train option")
    scenario_type: str = Field(..., description="Type of dilemma scenario")
    expected_optimal_split_junction: str = Field(..., description="Expected junction code")
    min_acceptable_buffer_mins: int = Field(..., ge=10, description="Minimum buffer in minutes")
    max_acceptable_layover_mins: int = Field(..., ge=60, description="Maximum layover in minutes")
    difficulty: str = Field(..., description="Difficulty level ('standard' or 'edge_case')")
    notes: str = Field("", description="Explanatory notes for the test case")


def parse_time_to_minutes(time_str: str) -> int:
    """Convert HH:MM string to minutes past midnight."""
    hh, mm = map(int, time_str.split(":"))
    return hh * 60 + mm


class BaselineSolver:
    """Naive timetable graph search solver (heuristic baseline).
    Ignores strict layover buffer limits and average delay risks.
    """

    def __init__(self, train_network: Dict[str, Any]):
        self.stations = {s["code"]: s for s in train_network.get("stations", [])}
        self.trains = train_network.get("trains", [])

    def solve(self, tc: TestCase) -> Tuple[Optional[SplitItinerary], bool, bool]:
        """Returns (itinerary, is_hallucination, is_operationally_feasible)."""
        origin = tc.origin
        dest = tc.destination

        first_legs = [t for t in self.trains if t["src_station"] == origin]
        second_legs = [t for t in self.trains if t["dest_station"] == dest]

        for t1 in first_legs:
            junc = t1["dest_station"]
            matching_t2 = [t for t in second_legs if t["src_station"] == junc]

            for t2 in matching_t2:
                arr1 = parse_time_to_minutes(t1["arrival_time"])
                dep2 = parse_time_to_minutes(t2["departure_time"])

                if dep2 > arr1:
                    layover = dep2 - arr1
                else:
                    layover = (dep2 + 1440) - arr1

                if 0 < layover <= tc.max_acceptable_layover_mins:
                    leg1 = Leg(
                        train=TrainSchedule(**{k: v for k, v in t1.items() if k in TrainSchedule.model_fields}),
                        from_station=t1["src_station"],
                        to_station=t1["dest_station"],
                        dep_time=t1["departure_time"],
                        arr_time=t1["arrival_time"],
                        class_type=tc.preferred_class if tc.preferred_class in t1["classes"] else t1["classes"][0],
                        availability_status=t1.get("availability_status", "AVAILABLE-0010"),
                        confirmation_prob=t1.get("confirmation_prob", 0.85)
                    )
                    leg2 = Leg(
                        train=TrainSchedule(**{k: v for k, v in t2.items() if k in TrainSchedule.model_fields}),
                        from_station=t2["src_station"],
                        to_station=t2["dest_station"],
                        dep_time=t2["departure_time"],
                        arr_time=t2["arrival_time"],
                        class_type=tc.preferred_class if tc.preferred_class in t2["classes"] else t2["classes"][0],
                        availability_status=t2.get("availability_status", "AVAILABLE-0010"),
                        confirmation_prob=t2.get("confirmation_prob", 0.85)
                    )

                    is_feasible = (layover >= tc.min_acceptable_buffer_mins) and (layover - t1.get("avg_delay_mins", 0) >= 15)

                    duration = (parse_time_to_minutes(t2["arrival_time"]) + t2.get("day_offset", 0) * 1440) - parse_time_to_minutes(t1["departure_time"])
                    if duration < 0:
                        duration += 1440

                    itinerary = SplitItinerary(
                        route_id=f"BASE-{tc.id}-{junc}",
                        origin=origin,
                        destination=dest,
                        legs=[leg1, leg2],
                        total_duration_mins=duration,
                        layover_buffer_mins=layover,
                        is_operationally_feasible=is_feasible,
                        feasibility_notes="Naive graph join path found." if is_feasible else f"Naive path accepted despite layover {layover}m or delay risk.",
                        overall_confirmation_prob=round(leg1.confirmation_prob * leg2.confirmation_prob, 2)
                    )
                    return itinerary, False, is_feasible

        return None, False, False


class AgentSolver:
    """Agentic split-journey planner solver.
    Enforces minimum transfer buffer limits, delay safety margins, and verification.
    """

    def __init__(self, train_network: Dict[str, Any]):
        self.stations = {s["code"]: s for s in train_network.get("stations", [])}
        self.trains = train_network.get("trains", [])

    def solve(self, tc: TestCase) -> Tuple[Optional[SplitItinerary], bool, bool]:
        """Returns (itinerary, is_hallucination, is_operationally_feasible)."""
        origin = tc.origin
        dest = tc.destination

        first_legs = [t for t in self.trains if t["src_station"] == origin]
        second_legs = [t for t in self.trains if t["dest_station"] == dest]

        best_itinerary = None
        best_score = -1.0

        for t1 in first_legs:
            junc = t1["dest_station"]
            matching_t2 = [t for t in second_legs if t["src_station"] == junc]

            for t2 in matching_t2:
                arr1 = parse_time_to_minutes(t1["arrival_time"])
                dep2 = parse_time_to_minutes(t2["departure_time"])

                if dep2 > arr1:
                    layover = dep2 - arr1
                else:
                    layover = (dep2 + 1440) - arr1

                if layover < tc.min_acceptable_buffer_mins:
                    continue

                t1_delay = t1.get("avg_delay_mins", 0)
                effective_buffer = layover - t1_delay
                if effective_buffer < 15:
                    continue

                if layover > tc.max_acceptable_layover_mins:
                    continue

                prob1 = t1.get("confirmation_prob", 0.85)
                prob2 = t2.get("confirmation_prob", 0.85)
                overall_prob = round(prob1 * prob2, 2)

                leg1 = Leg(
                    train=TrainSchedule(**{k: v for k, v in t1.items() if k in TrainSchedule.model_fields}),
                    from_station=t1["src_station"],
                    to_station=t1["dest_station"],
                    dep_time=t1["departure_time"],
                    arr_time=t1["arrival_time"],
                    class_type=tc.preferred_class if tc.preferred_class in t1["classes"] else t1["classes"][0],
                    availability_status=t1.get("availability_status", "AVAILABLE-0010"),
                    confirmation_prob=prob1
                )
                leg2 = Leg(
                    train=TrainSchedule(**{k: v for k, v in t2.items() if k in TrainSchedule.model_fields}),
                    from_station=t2["src_station"],
                    to_station=t2["dest_station"],
                    dep_time=t2["departure_time"],
                    arr_time=t2["arrival_time"],
                    class_type=tc.preferred_class if tc.preferred_class in t2["classes"] else t2["classes"][0],
                    availability_status=t2.get("availability_status", "AVAILABLE-0010"),
                    confirmation_prob=prob2
                )

                duration = (parse_time_to_minutes(t2["arrival_time"]) + t2.get("day_offset", 0) * 1440) - parse_time_to_minutes(t1["departure_time"])
                if duration < 0:
                    duration += 1440

                itinerary = SplitItinerary(
                    route_id=f"AGENT-{tc.id}-{junc}",
                    origin=origin,
                    destination=dest,
                    legs=[leg1, leg2],
                    total_duration_mins=duration,
                    layover_buffer_mins=layover,
                    is_operationally_feasible=True,
                    feasibility_notes=f"Feasible split via {junc}. Layover {layover}m >= min {tc.min_acceptable_buffer_mins}m with safe delay margin ({effective_buffer}m).",
                    overall_confirmation_prob=overall_prob
                )

                if overall_prob > best_score:
                    best_score = overall_prob
                    best_itinerary = itinerary

        if best_itinerary:
            return best_itinerary, False, True

        return None, False, False


class BenchmarkEvaluator:
    """Automated benchmark evaluator for RailRouteAgent."""

    def __init__(self, test_cases_path: Path = TEST_CASES_JSON, network_path: Path = TRAIN_NETWORK_JSON):
        self.test_cases_path = test_cases_path
        self.network_path = network_path
        self.test_cases: List[TestCase] = []
        self.train_network: Dict[str, Any] = {}

    def load_and_validate_test_cases(self) -> List[TestCase]:
        """Loads and validates test cases against TestCase schema."""
        if not self.test_cases_path.exists():
            raise FileNotFoundError(f"Test cases file not found at {self.test_cases_path}")

        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        validated_cases = []
        for idx, item in enumerate(raw_data):
            try:
                tc = TestCase(**item)
                validated_cases.append(tc)
            except ValidationError as e:
                raise ValueError(f"Schema validation failed for test case index {idx} ({item.get('id', 'Unknown')}): {e}")

        self.test_cases = validated_cases
        return self.test_cases

    def load_network(self):
        """Load synthetic timetable network data."""
        if not self.network_path.exists():
            raise FileNotFoundError(f"Train network file not found at {self.network_path}")

        with open(self.network_path, "r", encoding="utf-8") as f:
            self.train_network = json.load(f)

    def verify_llm_itinerary(self, tc: TestCase, itinerary: Optional[SplitItinerary]) -> Tuple[bool, bool]:
        """Strictly verifies LLM output for Hallucination and Operational Feasibility.
        Returns: (is_hallucination, is_feasible)
        """
        if itinerary is None:
            return False, False

        valid_train_nos = {t["train_no"] for t in self.train_network.get("trains", [])}
        train_map = {t["train_no"]: t for t in self.train_network.get("trains", [])}

        # Check for Hallucination: train numbers must exist in dataset
        is_hallucination = False
        for leg in itinerary.legs:
            if leg.train.train_no not in valid_train_nos:
                is_hallucination = True
                break

        if is_hallucination:
            return True, False

        # Verify operational feasibility constraints
        # 1. Layover buffer must be >= min_acceptable_buffer_mins
        if itinerary.layover_buffer_mins < tc.min_acceptable_buffer_mins:
            return False, False

        # 2. Check cascading delay safety margin for leg 1
        leg1_train_no = itinerary.legs[0].train.train_no
        t1_data = train_map.get(leg1_train_no, {})
        avg_delay = t1_data.get("avg_delay_mins", 0)

        effective_buffer = itinerary.layover_buffer_mins - avg_delay
        if effective_buffer < 15:
            return False, False

        return False, True

    def calculate_metrics(self, solver_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics for a set of solver results."""
        total = len(solver_results)
        if total == 0:
            return {
                "viable_route_found_rate": 0.0,
                "operational_feasibility_pass_rate": 0.0,
                "hallucination_rate": 0.0,
                "avg_confirmation_prob": 0.0
            }

        viable_count = sum(1 for r in solver_results if r["itinerary"] is not None)
        feasible_count = sum(1 for r in solver_results if r["itinerary"] is not None and r["is_feasible"])
        hallucination_count = sum(1 for r in solver_results if r["is_hallucination"])

        probs = [r["itinerary"].overall_confirmation_prob for r in solver_results if r["itinerary"] is not None]
        avg_prob = sum(probs) / len(probs) if probs else 0.0

        return {
            "viable_route_found_rate": round((viable_count / total) * 100.0, 2),
            "operational_feasibility_pass_rate": round((feasible_count / total) * 100.0, 2),
            "hallucination_rate": round((hallucination_count / total) * 100.0, 2),
            "avg_confirmation_prob": round(avg_prob, 4)
        }

    def run_baseline_llm(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute Zero-Shot LLM Baseline Agent (Iteration 0) and log trajectories."""
        self.load_and_validate_test_cases()
        self.load_network()

        baseline_agent = ZeroShotBaselineAgent(self.train_network)
        baseline_results = []
        trajectories = []

        for tc in self.test_cases:
            itin, prompt, raw_response = baseline_agent.solve(
                origin=tc.origin,
                destination=tc.destination,
                preferred_class=tc.preferred_class,
                date_offset=tc.travel_date_offset_days
            )

            is_hallu, is_feas = self.verify_llm_itinerary(tc, itin)

            baseline_results.append({
                "tc_id": tc.id,
                "itinerary": itin,
                "is_hallucination": is_hallu,
                "is_feasible": is_feas
            })

            trajectories.append({
                "test_case_id": tc.id,
                "scenario_type": tc.scenario_type,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_itinerary": itin.model_dump() if itin else None,
                "is_hallucination": is_hallu,
                "is_feasible": is_feas
            })

        # Save baseline trajectories log
        TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        traj_file = TRAJECTORIES_DIR / "baseline_trajectories.json"
        with open(traj_file, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2)

        metrics = self.calculate_metrics(baseline_results)
        return metrics, trajectories

    def run_evaluation(self, mode: str = "compare") -> Dict[str, Any]:
        """Executes evaluation across requested mode: 'baseline', 'heuristic', 'agent', or 'compare'."""
        self.load_and_validate_test_cases()
        self.load_network()

        heuristic_solver = BaselineSolver(self.train_network)
        agent_solver = AgentSolver(self.train_network)

        h_results, a_results = [], []
        detailed_logs = []

        # Run Zero-Shot Baseline LLM if mode is baseline or compare
        b_llm_metrics, b_trajectories = None, None
        if mode in ("baseline", "compare"):
            b_llm_metrics, b_trajectories = self.run_baseline_llm()

        for tc in self.test_cases:
            h_itin, h_hallu, h_feas = heuristic_solver.solve(tc)
            a_itin, a_hallu, a_feas = agent_solver.solve(tc)

            h_results.append({"tc_id": tc.id, "itinerary": h_itin, "is_hallucination": h_hallu, "is_feasible": h_feas})
            a_results.append({"tc_id": tc.id, "itinerary": a_itin, "is_hallucination": a_hallu, "is_feasible": a_feas})

            detailed_logs.append({
                "test_case_id": tc.id,
                "difficulty": tc.difficulty,
                "scenario_type": tc.scenario_type,
                "heuristic_baseline": {
                    "found": h_itin is not None,
                    "is_feasible": h_feas,
                    "route_id": h_itin.route_id if h_itin else None,
                    "layover_buffer_mins": h_itin.layover_buffer_mins if h_itin else None,
                },
                "agent": {
                    "found": a_itin is not None,
                    "is_feasible": a_feas,
                    "route_id": a_itin.route_id if a_itin else None,
                    "layover_buffer_mins": a_itin.layover_buffer_mins if a_itin else None,
                }
            })

        h_metrics = self.calculate_metrics(h_results)
        a_metrics = self.calculate_metrics(a_results)

        evaluation_output = {
            "summary_metrics": {
                "zero_shot_llm_baseline": b_llm_metrics or self.calculate_metrics([]),
                "heuristic_baseline": h_metrics,
                "agent": a_metrics
            },
            "detailed_case_results": detailed_logs
        }

        return evaluation_output

    def display_rich_summary(self, eval_results: Dict[str, Any]):
        """Render beautiful summary table comparing Zero-Shot Baseline vs Heuristic vs Agent."""
        console = Console()
        b_llm = eval_results["summary_metrics"].get("zero_shot_llm_baseline", {})
        h_m = eval_results["summary_metrics"]["heuristic_baseline"]
        a_m = eval_results["summary_metrics"]["agent"]

        table = Table(title="[RailRouteAgent] Benchmark Evaluation Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold white", min_width=35, no_wrap=True)
        table.add_column("Zero-Shot LLM (Iter 0)", justify="center", style="yellow", min_width=22, no_wrap=True)
        table.add_column("Agent Solution", justify="center", style="bold green", min_width=20, no_wrap=True)
        table.add_column("Improvement", justify="center", style="bold magenta", min_width=18, no_wrap=True)

        # Viable Route Found Rate
        diff_viable = a_m["viable_route_found_rate"] - b_llm.get("viable_route_found_rate", 0.0)
        table.add_row(
            "Viable Route Found Rate (%)",
            f"{b_llm.get('viable_route_found_rate', 0.0):.1f}%",
            f"{a_m['viable_route_found_rate']:.1f}%",
            f"{diff_viable:+.1f}%"
        )

        # Operational Feasibility Pass Rate
        diff_feas = a_m["operational_feasibility_pass_rate"] - b_llm.get("operational_feasibility_pass_rate", 0.0)
        table.add_row(
            "Operational Feasibility Pass Rate (%)",
            f"{b_llm.get('operational_feasibility_pass_rate', 0.0):.1f}%",
            f"{a_m['operational_feasibility_pass_rate']:.1f}%",
            f"[bold green]{diff_feas:+.1f}%[/bold green]"
        )

        # Hallucination / Invalid Connection Rate
        diff_hallu = a_m["hallucination_rate"] - b_llm.get("hallucination_rate", 0.0)
        table.add_row(
            "Hallucination / Invalid Connection Rate (%)",
            f"{b_llm.get('hallucination_rate', 0.0):.1f}%",
            f"{a_m['hallucination_rate']:.1f}%",
            f"{diff_hallu:+.1f}%"
        )

        # Average Confirmation Probability
        diff_prob = a_m["avg_confirmation_prob"] - b_llm.get("avg_confirmation_prob", 0.0)
        table.add_row(
            "Average Confirmation Probability",
            f"{b_llm.get('avg_confirmation_prob', 0.0):.4f}",
            f"{a_m['avg_confirmation_prob']:.4f}",
            f"{diff_prob:+.4f}"
        )

        console.print(Panel(table, border_style="green", title="Iteration Evaluation Completed"))

    def save_results(self, eval_results: Dict[str, Any], output_path: Path = LOGS_DIR / "benchmark_results.json"):
        """Save benchmark results to logs directory."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RailRouteAgent Benchmark Scoring Engine & Evaluator")
    parser.add_argument("--dry-run", action="store_true", help="Validate test cases schema without running solver evaluation")
    parser.add_argument("--mode", choices=["baseline", "heuristic", "agent", "compare"], default="compare", help="Evaluation mode")
    parser.add_argument("--test-cases", type=Path, default=TEST_CASES_JSON, help="Path to test cases JSON file")
    parser.add_argument("--output", type=Path, default=LOGS_DIR / "benchmark_results.json", help="Path to write evaluation results")

    args = parser.parse_args()
    evaluator = BenchmarkEvaluator(test_cases_path=args.test_cases)

    if args.dry_run:
        console = Console()
        try:
            cases = evaluator.load_and_validate_test_cases()
            console.print(Panel(
                f"[bold green]Schema validation SUCCESS![/bold green]\n"
                f"Successfully validated [bold cyan]{len(cases)}[/bold cyan] benchmark test cases from {args.test_cases}.",
                title="[RailRouteAgent] Dry-Run Validation Passed",
                border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                f"[bold red]Schema validation FAILED![/bold red]\nError: {e}",
                title="[RailRouteAgent] Dry-Run Validation Error",
                border_style="red"
            ))
            raise SystemExit(1)
    else:
        results = evaluator.run_evaluation(mode=args.mode)
        evaluator.display_rich_summary(results)
        evaluator.save_results(results, output_path=args.output)


if __name__ == "__main__":
    main()
