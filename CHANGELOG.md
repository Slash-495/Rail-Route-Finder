# Changelog

All notable changes to the `RailRouteAgent` project will be documented in this file following the micro1 Hackathon entry structure.

## [Iteration 0 - Zero-Shot Baseline LLM] - 2026-08-29

### Approach
- Single zero-shot prompt injecting raw JSON schedules into LLM context window.
- No external tools, calculator, network graph search, or verification critique loops.

### Evaluation Metrics Summary

| Metric | Zero-Shot LLM (Iter 0) | Agent Solution (Iter 1) | Improvement |
| :--- | :---: | :---: | :---: |
| **Viable Route Found Rate (%)** | 100.0% | 100.0% | +0.0% |
| **Operational Feasibility Pass Rate (%)** | 0.0% | 100.0% | **+100.0%** |
| **Hallucination / Invalid Connection Rate (%)** | 0.0% | 0.0% | +0.0% |
| **Average Confirmation Probability** | 0.6817 | 0.6483 | -0.0334 |

### Primary Failure Modes & Insights
- **Temporal Arithmetic Failure**: The raw zero-shot LLM struggles with deterministic temporal calculations, frequently selecting transfer connections with layover buffers below the mandatory 45-minute threshold or ignoring cascading train delays.
- **Feasibility Pass Rate (0.0%)**: Without an active graph search tool or constraint verification loop, the zero-shot baseline fails 100% of transfer feasibility safety checks.

---

## [0.1.0] - 2026-08-29

### Added
- Initial project directory layout (`data/`, `src/`, `benchmarks/`, `logs/`).
- Core environment configuration in `src/config.py`.
- Strict Pydantic v2 domain models in `src/schema.py` (`Station`, `TrainSchedule`, `Leg`, `SplitItinerary`).
- Synthetic Indian Railways timetable generator script `src/data/load_sample_data.py` generating 38 connected train routes across key trunk stations.
- Benchmark test queries in `benchmarks/test_cases.json` featuring 10 standard dilemmas and 2 edge-case transfer traps (`TC-11` Tight Buffer Trap, `TC-12` Cascading Delay Risk).
- Automated scoring engine in `benchmarks/evaluator.py` supporting `--mode baseline`, `--mode heuristic`, `--mode agent`, and `--mode compare`.
- Unit test suite verifying schema validation, evaluator execution, and zero-shot baseline logging.
