# 🚆 RailRouteAgent: Agentic IRCTC Split-Journey Planner

`RailRouteAgent` is an intelligent, agentic Indian Railways split-journey planner designed to uncover high-probability split itineraries (end-to-end multi-leg train connections) when direct tickets are sold out, on waitlist, or sub-optimal.

## 📁 Repository Structure

```
rail-route-finder/
├── data/
│   ├── raw/                  # Raw IRCTC / timetable exports
│   └── processed/            # Serialized timetable graph network (train_network.json)
├── src/
│   ├── __init__.py
│   ├── config.py             # System paths, constants, and environment configurations
│   ├── schema.py             # Pydantic v2 domain schemas (Station, TrainSchedule, Leg, SplitItinerary)
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_sample_data.py # Synthetic timetable graph generator script
│   ├── tools/                # Agent tools (routing, feasibility check, IRCTC lookup)
│   ├── agents/               # Multi-agent coordinator and reasoning subagents
│   └── utils/                # Date/time, graph algorithms, and formatting helpers
├── benchmarks/
│   └── test_cases.json       # Benchmark search queries for split journey evaluation
├── logs/
│   └── trajectories/         # Execution logs and agent trajectory traces
├── README.md
├── CHANGELOG.md
└── requirements.txt
```

## ⚙️ Installation & Setup

1. **Clone & Virtual Environment Setup**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Synthetic Timetable Dataset**:
   ```bash
   python -m src.data.load_sample_data
   ```

4. **Run Unit Tests**:
   ```bash
   pytest
   ```

## 🧩 Schema Architecture

- **`Station`**: Represents station nodes, codes, zones, and platform counts.
- **`TrainSchedule`**: Train route timing, departure/arrival schedules, delay metrics, and classes.
- **`Leg`**: Specific leg within a journey with IRCTC availability status and calculated confirmation probabilities.
- **`SplitItinerary`**: End-to-end split route with feasibility validation, layover buffer metrics, and overall confirmation probability.
