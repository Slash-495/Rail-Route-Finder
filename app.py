"""
RailRouteAgent Streamlit Dashboard & Trajectory Viewer
Multi-Agent Graph Search & Risk Audit Engine for Indian Railways Split Journeys.
"""
import os
os.makedirs("logs/trajectories", exist_ok=True)
import datetime
import json
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

from src.agents import VerifierAgent, RankingAgent
from src.config import LOGS_DIR, TRAJECTORIES_DIR, BENCHMARK_RESULTS_JSON

st.write(os.listdir("logs/trajectories"))
st.set_page_config(
    page_title="RailRouteAgent - Split Journey Planner",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header
st.title("🚄 RailRouteAgent")
st.subheader("Multi-Agent Graph Search & Risk Audit Engine for Indian Railways Split Journeys")
st.markdown(
    "Uncovers operationally safe, high-probability 2-leg transfer itineraries when direct train tickets are sold out or heavily waitlisted."
)

query_tab = st.query_params.get("tab", "demo")
if query_tab == "viewer":
    tab2, tab1 = st.tabs(["📜 Trajectory Viewer", "🚆 Live Demo"])
else:
    tab1, tab2 = st.tabs(["🚆 Live Demo", "📜 Trajectory Viewer"])

# -----------------------------------------------------------------------------
# TAB 1: Live Demo
# -----------------------------------------------------------------------------
with tab1:
    st.sidebar.header("🗺️ Journey Parameters")
    
    with st.sidebar.form("route_form"):
        origin_input = st.text_input("Origin Station Code", value="NDLS", help="e.g. NDLS (New Delhi)")
        dest_input = st.text_input("Destination Station Code", value="MAO", help="e.g. MAO (Madgaon, Goa)")
        date_input = st.date_input("Travel Date", value=datetime.date(2026, 9, 15))
        submitted = st.form_submit_button("Find Verified Routes", type="primary")

    if submitted or st.session_state.get("ran_query"):
        if submitted:
            st.session_state["ran_query"] = True
            st.session_state["origin"] = origin_input
            st.session_state["dest"] = dest_input
            st.session_state["date"] = str(date_input)

        orig = st.session_state.get("origin", "NDLS").strip().upper()
        dest = st.session_state.get("dest", "MAO").strip().upper()
        travel_date = st.session_state.get("date", "2026-09-15")

        with st.spinner("Agents are traversing the graph and auditing risk..."):
            verifier = VerifierAgent()
            ranker = RankingAgent()
            verified_routes = verifier.verify_and_refine(orig, dest, travel_date)

        if not verified_routes:
            st.error("❌ No operationally feasible split-journey routes found matching the mandatory safety criteria.")
        else:
            report_markdown = ranker.rank_and_summarize(verified_routes)
            st.markdown(report_markdown)
            
            st.divider()
            st.markdown("### 🛡️ IRCTC Booking Safety Gate (Consequential Action Safeguard)")
            if st.button("Simulate Booking (Sandbox)", type="secondary"):
                st.success("✅ [SANDBOX] Booking simulation successful. No actual transaction occurred.")
    else:
        st.info("👈 Enter origin, destination, and travel date in the sidebar form and click **Find Verified Routes** to run the multi-agent planning & reflection pipeline.")

# -----------------------------------------------------------------------------
# TAB 2: Trajectory Viewer
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("## 📜 Execution Trajectory Viewer")
    st.markdown("Inspect multi-agent internal reasoning, tool calls, verifier reflection critiques, and telemetry event logs step-by-step.")

    trajectories_dir = TRAJECTORIES_DIR
    if not trajectories_dir.exists():
        st.warning(f"⚠️ Trajectories directory not found at `{trajectories_dir}`.")
    else:
        json_files = sorted(list(trajectories_dir.glob("*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            st.warning("⚠️ No trajectory JSON files found in `logs/trajectories/`.")
        else:
            file_options = {f.name: f for f in json_files}
            default_file = "run_NDLS_MAO_20260915.json" if "run_NDLS_MAO_20260915.json" in file_options else list(file_options.keys())[0]
            default_idx = list(file_options.keys()).index(default_file)
            
            selected_file_name = st.selectbox("Select Trajectory Trace File", options=list(file_options.keys()), index=default_idx)
            selected_path = file_options[selected_file_name]

            try:
                with open(selected_path, "r", encoding="utf-8") as f:
                    events = json.load(f)

                if not isinstance(events, list):
                    st.error("Invalid trajectory JSON format. Expected an array of event objects.")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Steps", len(events))
                    col2.metric("Planner Events", sum(1 for e in events if e.get("agent") == "Planner"))
                    col3.metric("Verifier Audits", sum(1 for e in events if e.get("agent") == "Verifier"))
                    col4.metric("Tool Calls", sum(1 for e in events if e.get("event_type") == "tool_call"))

                    st.divider()
                    st.markdown("### Step-by-Step Event Trajectory")

                    for idx, event in enumerate(events):
                        agent = event.get("agent", "System")
                        event_type = event.get("event_type", "unknown")
                        timestamp = event.get("timestamp", "")
                        payload = event.get("payload", {})

                        badge_color = {
                            "Planner": "🟦",
                            "Verifier": "🟥",
                            "Ranking": "🟩",
                            "System": "🟧"
                        }.get(agent, "⚪")

                        expander_title = f"{badge_color} Step {idx + 1}: [{agent}] — {event_type.upper().replace('_', ' ')} ({timestamp[:19]})"

                        with st.expander(expander_title, expanded=(idx in [0, 1, 2, 3] or event_type in ["reflection_feedback", "human_checkpoint"])):
                            if event_type == "prompt":
                                st.markdown(f"**Agent**: `{agent}`")
                                if isinstance(payload, dict):
                                    if "prompt" in payload:
                                        st.markdown(f"**User/Input Prompt**:\n```text\n{payload['prompt']}\n```")
                                    if "system_prompt" in payload:
                                        st.markdown(f"**System Instruction**:\n> {payload['system_prompt']}")
                                    if "origin" in payload:
                                        st.json(payload)
                                else:
                                    st.write(payload)

                            elif event_type == "tool_call":
                                st.markdown(f"**Tool Invoked**: `{payload.get('tool')}`")
                                if "args" in payload:
                                    st.markdown("**Arguments**:")
                                    st.json(payload["args"])

                            elif event_type == "tool_response":
                                st.markdown(f"**Tool Response Source**: `{payload.get('tool', agent)}`")
                                st.json(payload)

                            elif event_type == "reflection_feedback":
                                st.warning(f"⚠️ **Verifier Critique Feedback**:\n\n{payload.get('critique', payload)}")
                                if "iteration" in payload:
                                    st.caption(f"Reflection Loop Iteration {payload['iteration']}")

                            elif event_type == "human_checkpoint":
                                st.info(f"🛡️ **Human-in-the-Loop Safeguard Action**:\n\nAction: `{payload.get('action')}` | User Input: `{payload.get('user_input')}`")
                                st.json(payload)

                            else:
                                st.json(payload)

            except Exception as e:
                st.error(f"Error reading trajectory file `{selected_file_name}`: {e}")
