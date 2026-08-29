# Changelog

All notable changes to the `RailRouteAgent` project will be documented in this file.

## [0.1.0] - 2026-08-29

### Added
- Initial project directory layout (`data/`, `src/`, `benchmarks/`, `logs/`).
- Core environment configuration in `src/config.py`.
- Strict Pydantic v2 domain models in `src/schema.py` (`Station`, `TrainSchedule`, `Leg`, `SplitItinerary`).
- Synthetic Indian Railways timetable generator script `src/data/load_sample_data.py` generating 25+ connected train routes across key trunk stations.
- Benchmark test queries in `benchmarks/test_cases.json`.
- Unit test suite verifying schema validation and dataset integrity.
