"""
RankingAgent module for RailRouteAgent.
Synthesizes operationally verified routes into ranked, human-readable travel recommendations.
"""

import json
from typing import List, Dict, Any, Optional

from src.config import GEMINI_API_KEY
from src.tools.ml_scorer import predict_pnr_confirmation

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


RANKING_SYSTEM_PROMPT = (
    "You are an expert Indian Railways Travel Advisor. You will be given a list of operationally verified split-journey routes. "
    "Your job is to analyze the trade-offs between 'Total Travel Time', 'Layover Safety', and 'Ticket Confirmation Probability'. "
    "Rank the routes from best to worst. "
    "Output a synthesized Markdown summary presenting the Top 3 options to the user. For each option, clearly explain *why* it is "
    "ranked there (e.g., 'Best Overall', 'Safest Buffer', 'Highest Confirmation Chance')."
)


class RankingAgent:
    """Synthesis agent that enriches verified routes with ML PNR probabilities and generates human-readable Markdown recommendations."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.trajectory: List[Dict[str, Any]] = []

    def rank_and_summarize(self, verified_routes: List[Dict[str, Any]], days_to_journey: int = 5) -> str:
        """Enriches verified routes with PNR confirmation scores and synthesizes a ranked Markdown travel report.

        Args:
            verified_routes: List of operationally verified SplitItinerary model dumps or dictionaries.
            days_to_journey: Days remaining before departure (default 5).

        Returns:
            str: Synthesized Markdown report presenting ranked split-journey recommendations.
        """
        self.trajectory = []

        if not verified_routes:
            fallback_msg = "### RailRouteAgent Journey Advisory\n\nNo operationally feasible split-journey routes found matching the mandatory safety criteria."
            self.trajectory.append({"step": "empty_input", "output": fallback_msg})
            return fallback_msg

        # 1. DATA ENRICHMENT: Calculate leg and overall confirmation probabilities
        enriched_routes = self._enrich_routes_with_pnr_probs(verified_routes, days_to_journey)

        # Sort routes by composite score (confirmation prob desc, layover buffer desc, duration asc)
        enriched_routes.sort(
            key=lambda r: (
                r.get("overall_confirmation_prob", 0.0),
                r.get("layover_buffer_mins", 0),
                -r.get("total_duration_mins", 9999)
            ),
            reverse=True
        )

        input_payload = json.dumps(enriched_routes, indent=2)

        self.trajectory.append({
            "step": "input_payload",
            "routes_count": len(enriched_routes),
            "payload": enriched_routes
        })

        # 2. LLM SYNTHESIS or FALLBACK
        if GEMINI_API_KEY and HAS_GENAI:
            markdown_report = self._generate_llm_summary(input_payload)
            if markdown_report:
                self.trajectory.append({"step": "llm_output", "markdown": markdown_report})
                return markdown_report

        # Deterministic fallback renderer
        markdown_report = self._render_fallback_markdown(enriched_routes)
        self.trajectory.append({"step": "fallback_output", "markdown": markdown_report})
        return markdown_report

    def _enrich_routes_with_pnr_probs(self, routes: List[Dict[str, Any]], days_to_journey: int) -> List[Dict[str, Any]]:
        """Calculates leg confirmation probabilities using predict_pnr_confirmation tool."""
        enriched = []
        for r in routes:
            r_copy = dict(r)
            legs = r_copy.get("legs", [])
            probs = []

            for leg in legs:
                status = leg.get("availability_status", "AVAILABLE-0010")
                class_type = leg.get("class_type", "3A")
                prob = predict_pnr_confirmation(status, class_type, days_to_journey)
                leg["confirmation_prob"] = prob
                probs.append(prob)

            if probs:
                overall = round(sum(probs) / len(probs), 2)
            else:
                overall = r_copy.get("overall_confirmation_prob", 0.85)

            r_copy["overall_confirmation_prob"] = overall
            enriched.append(r_copy)

        return enriched

    def _generate_llm_summary(self, input_payload: str) -> Optional[str]:
        """Queries Gemini LLM for Markdown report generation."""
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"Operationally Verified Split Routes Payload:\n{input_payload}\n\nSynthesize the Top 3 options in clear Markdown format."

            res = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RANKING_SYSTEM_PROMPT
                )
            )

            return res.text
        except Exception:
            return None

    def _render_fallback_markdown(self, routes: List[Dict[str, Any]]) -> str:
        """Renders polished, professional Markdown summary report."""
        lines = []
        lines.append("## RailRouteAgent Split-Journey Recommendations\n")
        lines.append(f"Analyzed **{len(routes)} operationally verified split itineraries**. Below are the top ranked options:\n")

        badges = ["Option 1: Best Overall", "Option 2: Safest Buffer", "Option 3: Balanced Alternative"]

        for idx, r in enumerate(routes[:3]):
            badge = badges[idx] if idx < len(badges) else f"Option {idx + 1}"
            orig = r.get("origin", "N/A")
            dest = r.get("destination", "N/A")
            duration_hrs = round(r.get("total_duration_mins", 0) / 60, 1)
            layover_mins = r.get("layover_buffer_mins", 0)
            overall_prob = int(r.get("overall_confirmation_prob", 0.85) * 100)
            notes = r.get("feasibility_notes", "Operationally verified connection.")

            lines.append(f"### {badge} — Via Junction `{r.get('route_id', '').split('-')[-1]}`")
            lines.append(f"- **Journey Route**: `{orig}` -> `{dest}`")
            lines.append(f"- **Total Duration**: {duration_hrs} hours")
            lines.append(f"- **Transfer Layover Buffer**: {layover_mins} minutes")
            lines.append(f"- **Ticket Confirmation Probability**: **{overall_prob}%**")
            lines.append(f"- **Advisor Rationale**: {notes}\n")

            legs = r.get("legs", [])
            if len(legs) >= 2:
                t1 = legs[0].get("train", {})
                t2 = legs[1].get("train", {})
                lines.append("  | Leg | Train No & Name | From -> To | Departure | Arrival | Class |")
                lines.append("  |---|---|---|---|---|---|")
                lines.append(f"  | Leg 1 | {t1.get('train_no', 'N/A')} - {t1.get('train_name', 'N/A')} | `{legs[0].get('from_station')}` -> `{legs[0].get('to_station')}` | {legs[0].get('dep_time')} | {legs[0].get('arr_time')} | {legs[0].get('class_type', '3A')} |")
                lines.append(f"  | Leg 2 | {t2.get('train_no', 'N/A')} - {t2.get('train_name', 'N/A')} | `{legs[1].get('from_station')}` -> `{legs[1].get('to_station')}` | {legs[1].get('dep_time')} | {legs[1].get('arr_time')} | {legs[1].get('class_type', '3A')} |\n")

        lines.append("---\n*Generated by RailRouteAgent Multi-Agent Reflection Engine.*")
        return "\n".join(lines)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Returns trajectory log for Hackathon deliverable."""
        return self.trajectory
