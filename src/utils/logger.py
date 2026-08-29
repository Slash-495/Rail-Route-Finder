"""
Telemetry and Trajectory Logger for RailRouteAgent.
Records structured agent events, tool calls, reflection feedback loops, and human safety checkpoints.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import TRAJECTORIES_DIR


class TrajectoryLogger:
    """Singleton telemetry logger for recording Hackathon execution trajectories."""

    _instance: Optional["TrajectoryLogger"] = None

    def __new__(cls) -> "TrajectoryLogger":
        if cls._instance is None:
            cls._instance = super(TrajectoryLogger, cls).__new__(cls)
            cls._instance.events = []
        return cls._instance

    def log_event(self, agent: str, event_type: str, payload: Any):
        """Logs a single telemetry event.

        Args:
            agent: Acting agent name ("Planner", "Verifier", "Ranking", or "System").
            event_type: Category ("prompt", "tool_call", "tool_response", "reflection_feedback", "human_checkpoint").
            payload: Event data dictionary or string.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "event_type": event_type,
            "payload": payload
        }
        self.events.append(event)

    def export_trajectory(self, session_id: str) -> Path:
        """Exports accumulated trajectory events to logs/trajectories/run_{session_id}.json.

        Args:
            session_id: Unique session identifier string.

        Returns:
            Path: File path to exported JSON trajectory file.
        """
        TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        file_path = TRAJECTORIES_DIR / f"run_{session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
        return file_path

    def clear(self):
        """Clears accumulated trajectory events."""
        self.events = []


def get_logger() -> TrajectoryLogger:
    """Helper function to obtain global TrajectoryLogger singleton instance."""
    return TrajectoryLogger()
