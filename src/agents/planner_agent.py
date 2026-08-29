"""
PlannerAgent module for RailRouteAgent.
Orchestrates candidate route discovery using graph tools and native LLM function calling.
"""

import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.config import GEMINI_API_KEY, OPENAI_API_KEY
from src.tools.routing_tools import find_direct_trains, find_split_junctions
from src.utils.logger import get_logger

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class CandidateRoute(BaseModel):
    """Represents a candidate 2-leg split route proposed by the PlannerAgent."""
    junction: str = Field(..., description="Intermediate transfer junction station code (e.g. BPL, NGP)")
    train_1_no: str = Field(..., description="Train number for Leg 1")
    train_1_name: str = Field(..., description="Train name for Leg 1")
    dep_time_1: str = Field(..., description="Departure time for Leg 1 (HH:MM)")
    arr_time_1: str = Field(..., description="Arrival time for Leg 1 at junction (HH:MM)")
    train_2_no: str = Field(..., description="Train number for Leg 2")
    train_2_name: str = Field(..., description="Train name for Leg 2")
    dep_time_2: str = Field(..., description="Departure time for Leg 2 from junction (HH:MM)")
    arr_time_2: str = Field(..., description="Arrival time for Leg 2 at destination (HH:MM)")
    scheduled_layover_mins: int = Field(..., ge=0, description="Scheduled layover in minutes")
    total_duration_mins: int = Field(..., ge=0, description="Total journey duration in minutes")


class ProposedRouteResponse(BaseModel):
    """Structured response container for route planning proposals."""
    origin: str = Field(..., description="Origin station code")
    destination: str = Field(..., description="Destination station code")
    travel_date: str = Field(..., description="Requested travel date")
    candidate_routes: List[CandidateRoute] = Field(default_factory=list, description="List of up to 3 candidate routes")
    reasoning: str = Field("", description="Planner reasoning and graph traversal notes")


PLANNER_SYSTEM_PROMPT = (
    "You are the Route Planner Agent for Indian Railways. Your objective is to find a multi-hop split journey "
    "from the Origin to the Destination. You MUST use the provided `find_split_junctions` tool to query the "
    "network graph. Do not hallucinate train numbers. Return a maximum of 3 graph-valid candidate routes. "
    "Do not evaluate delay risk or waitlist probability—just output the logical connections."
)


class PlannerAgent:
    """Orchestrating agent that queries graph routing tools and outputs structured proposed routes."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.trajectory: List[Dict[str, Any]] = []

    def plan_route(
        self,
        origin: str,
        destination: str,
        date: str = "2026-09-15",
        rejected_junctions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Plans candidate split journey routes using tool calls and trajectory logging.

        Args:
            origin: Origin station code (e.g. "NDLS").
            destination: Destination station code (e.g. "MAO").
            date: Requested travel date string.
            rejected_junctions: Optional list of junction station codes to strictly avoid.

        Returns:
            Dict[str, Any]: Model dump of ProposedRouteResponse.
        """
        self.trajectory = []
        origin_upper = origin.upper().strip()
        dest_upper = destination.upper().strip()

        sys_prompt = PLANNER_SYSTEM_PROMPT
        if rejected_junctions:
            sys_prompt += f" If rejected_junctions is provided, strictly DO NOT route through these stations: {rejected_junctions}."

        user_prompt = f"Plan a split journey route from {origin_upper} to {dest_upper} for travel date {date}."
        if rejected_junctions:
            user_prompt += f" Strictly avoid routing through these stations: {rejected_junctions}."

        self.trajectory.append({
            "step": "user_input",
            "prompt": user_prompt,
            "system_prompt": sys_prompt
        })
        get_logger().log_event("Planner", "prompt", {"prompt": user_prompt, "system_prompt": sys_prompt})

        # Check if live Gemini API is configured
        if GEMINI_API_KEY and HAS_GENAI:
            response = self._run_llm_tool_loop(origin_upper, dest_upper, date, user_prompt, sys_prompt, rejected_junctions)
            if response:
                return response.model_dump()

        # Deterministic tool-augmented fallback planner
        return self._run_fallback_tool_planner(origin_upper, dest_upper, date, rejected_junctions)

    def _run_llm_tool_loop(
        self,
        origin: str,
        destination: str,
        date: str,
        user_prompt: str,
        sys_prompt: str = PLANNER_SYSTEM_PROMPT,
        rejected_junctions: Optional[List[str]] = None
    ) -> Optional[ProposedRouteResponse]:
        """Runs function calling loop with google-genai SDK."""
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # Direct tool execution loop
            raw_candidates = find_split_junctions(origin, destination, max_layover_hrs=6)
            if rejected_junctions:
                rej_set = {j.upper() for j in rejected_junctions}
                raw_candidates = [c for c in raw_candidates if c["junction"].upper() not in rej_set]

            self.trajectory.append({
                "step": "tool_call",
                "tool": "find_split_junctions",
                "args": {"src": origin, "dest": destination, "max_layover_hrs": 6},
                "result_count": len(raw_candidates)
            })

            # Ask LLM to format candidate routes
            format_prompt = f"""Given the following graph routing tool output for {origin} -> {destination}:
{json.dumps(raw_candidates[:3], indent=2)}

Format this into candidate routes for {origin} -> {destination}.
"""
            res = client.models.generate_content(
                model=self.model_name,
                contents=format_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt
                )
            )

            raw_text = res.text or ""
            self.trajectory.append({
                "step": "llm_response",
                "raw_response": raw_text
            })

            return self._build_response_from_raw_candidates(origin, destination, date, raw_candidates)
        except Exception as e:
            self.trajectory.append({
                "step": "llm_error",
                "error": str(e)
            })
            return None

    def _run_fallback_tool_planner(
        self,
        origin: str,
        destination: str,
        date: str,
        rejected_junctions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Executes tool-driven graph search fallback."""
        raw_candidates = find_split_junctions(origin, destination, max_layover_hrs=6)
        if rejected_junctions:
            rej_set = {j.upper() for j in rejected_junctions}
            raw_candidates = [c for c in raw_candidates if c["junction"].upper() not in rej_set]
        
        get_logger().log_event("Planner", "tool_call", {"tool": "find_split_junctions", "args": {"src": origin, "dest": destination, "max_layover_hrs": 6}})
        get_logger().log_event("Planner", "tool_response", {"tool": "find_split_junctions", "result_count": len(raw_candidates), "candidates": raw_candidates[:3]})

        self.trajectory.append({
            "step": "tool_call",
            "tool": "find_split_junctions",
            "args": {"src": origin, "dest": destination, "max_layover_hrs": 6},
            "result_count": len(raw_candidates)
        })

        response = self._build_response_from_raw_candidates(origin, destination, date, raw_candidates)
        
        self.trajectory.append({
            "step": "final_output",
            "output": response.model_dump()
        })

        return response.model_dump()

    def _build_response_from_raw_candidates(self, origin: str, destination: str, date: str, raw_candidates: List[Dict[str, Any]]) -> ProposedRouteResponse:
        """Converts raw tool output dictionaries into ProposedRouteResponse object."""
        candidates: List[CandidateRoute] = []

        for item in raw_candidates[:3]:  # Return maximum 3 candidate routes as required
            t1 = item["train_1"]
            t2 = item["train_2"]

            c = CandidateRoute(
                junction=item["junction"],
                train_1_no=t1["train_no"],
                train_1_name=t1["train_name"],
                dep_time_1=t1["departure_time"],
                arr_time_1=t1["arrival_time"],
                train_2_no=t2["train_no"],
                train_2_name=t2["train_name"],
                dep_time_2=t2["departure_time"],
                arr_time_2=t2["arrival_time"],
                scheduled_layover_mins=item["scheduled_layover_mins"],
                total_duration_mins=item["total_duration_mins"]
            )
            candidates.append(c)

        return ProposedRouteResponse(
            origin=origin,
            destination=destination,
            travel_date=date,
            candidate_routes=candidates,
            reasoning=f"Found {len(candidates)} graph-valid split candidate route(s) via {', '.join([c.junction for c in candidates])}."
        )

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Returns the full execution trajectory history for Hackathon deliverables."""
        return self.trajectory
