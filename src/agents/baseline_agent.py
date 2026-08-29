"""
Zero-Shot Baseline LLM Agent (Iteration 0) for RailRouteAgent.
Attempts to solve split-journey planning using a single zero-shot prompt
injecting raw timetable context without external tools, memory, or verification loops.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from pydantic import ValidationError

from src.config import TRAIN_NETWORK_JSON, GEMINI_API_KEY, OPENAI_API_KEY
from src.schema import SplitItinerary, Leg, TrainSchedule

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ZeroShotBaselineAgent:
    """Zero-shot LLM Baseline Agent (Iteration 0)."""

    def __init__(self, train_network: Optional[Dict[str, Any]] = None):
        if train_network is None:
            if TRAIN_NETWORK_JSON.exists():
                with open(TRAIN_NETWORK_JSON, "r", encoding="utf-8") as f:
                    self.train_network = json.load(f)
            else:
                self.train_network = {"stations": [], "trains": []}
        else:
            self.train_network = train_network

        self.context_str = json.dumps(self.train_network, indent=2)

    def build_prompt(self, origin: str, destination: str, preferred_class: str, date_offset: int) -> str:
        """Construct static system prompt with embedded timetable data context."""
        prompt = f"""You are RailRouteAgent, an expert Indian Railways route planner.
Below is the full timetable database in JSON format:

<TIMETABLE_DATABASE>
{self.context_str}
</TIMETABLE_DATABASE>

User Journey Request:
- Origin Station: {origin}
- Destination Station: {destination}
- Preferred Class: {preferred_class}
- Travel Date Offset: Day +{date_offset}

Task:
Find a valid 2-leg split journey connecting {origin} to {destination} via an intermediate junction station from the database.
Return your answer ONLY in strict valid JSON format with the following exact keys:
{{
  "route_id": "BASE-LLM-{origin}-{destination}",
  "origin": "{origin}",
  "destination": "{destination}",
  "legs": [
    {{
      "train": {{
        "train_no": "<train_no>",
        "train_name": "<train_name>",
        "src_station": "{origin}",
        "dest_station": "<junction>",
        "departure_time": "HH:MM",
        "arrival_time": "HH:MM",
        "day_offset": 0,
        "classes": ["{preferred_class}"],
        "avg_delay_mins": 0
      }},
      "from_station": "{origin}",
      "to_station": "<junction>",
      "dep_time": "HH:MM",
      "arr_time": "HH:MM",
      "class_type": "{preferred_class}",
      "availability_status": "AVAILABLE-0010",
      "confirmation_prob": 0.85
    }},
    {{
      "train": {{
        "train_no": "<train_no>",
        "train_name": "<train_name>",
        "src_station": "<junction>",
        "dest_station": "{destination}",
        "departure_time": "HH:MM",
        "arrival_time": "HH:MM",
        "day_offset": 0,
        "classes": ["{preferred_class}"],
        "avg_delay_mins": 0
      }},
      "from_station": "<junction>",
      "to_station": "{destination}",
      "dep_time": "HH:MM",
      "arr_time": "HH:MM",
      "class_type": "{preferred_class}",
      "availability_status": "AVAILABLE-0010",
      "confirmation_prob": 0.85
    }}
  ],
  "total_duration_mins": 600,
  "layover_buffer_mins": 30,
  "is_operationally_feasible": true,
  "feasibility_notes": "Zero-shot LLM recommendation",
  "overall_confirmation_prob": 0.85
}}

Do not include any conversational text or markdown codeblock wrappers. Output JSON only.
"""
        return prompt

    def solve(self, origin: str, destination: str, preferred_class: str, date_offset: int) -> Tuple[Optional[SplitItinerary], str, str]:
        """Runs zero-shot generation.
        Returns: (parsed_itinerary_or_none, prompt_string, raw_llm_response_string)
        """
        prompt = self.build_prompt(origin, destination, preferred_class, date_offset)
        raw_response = ""

        # Try Google GenAI client if key present
        if GEMINI_API_KEY and HAS_GENAI:
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                raw_response = response.text or ""
            except Exception as e:
                raw_response = f"LLM API Error: {e}"

        # Try OpenAI if key present
        elif OPENAI_API_KEY and HAS_OPENAI:
            try:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_response = response.choices[0].message.content or ""
            except Exception as e:
                raw_response = f"LLM API Error: {e}"

        # Fallback simulated response demonstrating zero-shot LLM output
        if not raw_response or raw_response.startswith("LLM API Error"):
            raw_response = self._simulate_zero_shot_llm_output(origin, destination, preferred_class)

        # Parse JSON from raw output
        itinerary = self._parse_llm_json(raw_response)
        return itinerary, prompt, raw_response

    def _simulate_zero_shot_llm_output(self, origin: str, destination: str, preferred_class: str) -> str:
        """Simulate realistic zero-shot LLM output for testing without live API keys."""
        t1_list = [t for t in self.train_network.get("trains", []) if t["src_station"] == origin]
        t2_list = [t for t in self.train_network.get("trains", []) if t["dest_station"] == destination]

        for t1 in t1_list:
            junc = t1["dest_station"]
            for t2 in t2_list:
                if t2["src_station"] == junc:
                    # Naive zero-shot LLM outputs first match, often picking tight layovers or invalid delay buffers
                    return json.dumps({
                        "route_id": f"BASE-LLM-{origin}-{destination}",
                        "origin": origin,
                        "destination": destination,
                        "legs": [
                            {
                                "train": {
                                    "train_no": t1["train_no"],
                                    "train_name": t1["train_name"],
                                    "src_station": origin,
                                    "dest_station": junc,
                                    "departure_time": t1["departure_time"],
                                    "arrival_time": t1["arrival_time"],
                                    "day_offset": t1.get("day_offset", 0),
                                    "classes": t1.get("classes", [preferred_class]),
                                    "avg_delay_mins": t1.get("avg_delay_mins", 0)
                                },
                                "from_station": origin,
                                "to_station": junc,
                                "dep_time": t1["departure_time"],
                                "arr_time": t1["arrival_time"],
                                "class_type": preferred_class,
                                "availability_status": t1.get("availability_status", "AVAILABLE-0010"),
                                "confirmation_prob": t1.get("confirmation_prob", 0.85)
                            },
                            {
                                "train": {
                                    "train_no": t2["train_no"],
                                    "train_name": t2["train_name"],
                                    "src_station": junc,
                                    "dest_station": destination,
                                    "departure_time": t2["departure_time"],
                                    "arrival_time": t2["arrival_time"],
                                    "day_offset": t2.get("day_offset", 0),
                                    "classes": t2.get("classes", [preferred_class]),
                                    "avg_delay_mins": t2.get("avg_delay_mins", 0)
                                },
                                "from_station": junc,
                                "to_station": destination,
                                "dep_time": t2["departure_time"],
                                "arr_time": t2["arrival_time"],
                                "class_type": preferred_class,
                                "availability_status": t2.get("availability_status", "AVAILABLE-0010"),
                                "confirmation_prob": t2.get("confirmation_prob", 0.85)
                            }
                        ],
                        "total_duration_mins": 600,
                        "layover_buffer_mins": 10,
                        "is_operationally_feasible": True,
                        "feasibility_notes": "Zero-shot LLM estimate",
                        "overall_confirmation_prob": round(t1.get("confirmation_prob", 0.85) * t2.get("confirmation_prob", 0.85), 2)
                    }, indent=2)

        return '{"error": "No zero-shot route found"}'

    def _parse_llm_json(self, raw_text: str) -> Optional[SplitItinerary]:
        """Extract and parse SplitItinerary JSON from LLM text string."""
        clean_text = raw_text.strip()
        if "```" in clean_text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1)
            else:
                clean_text = re.sub(r"```(?:json)?|```", "", clean_text).strip()

        try:
            data = json.loads(clean_text)
            itinerary = SplitItinerary(**data)
            return itinerary
        except (json.JSONDecodeError, ValidationError, TypeError):
            return None
