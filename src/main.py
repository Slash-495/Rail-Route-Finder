"""
Master Orchestrator CLI Application for RailRouteAgent.
Ties together PlannerAgent, VerifierAgent, and RankingAgent into a unified execution flow
with a Human-in-the-Loop booking simulation safety gate.
"""

import argparse
import sys
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agents import VerifierAgent, RankingAgent


def run_pipeline(origin: str, destination: str, date: str = "2026-09-15", interactive: bool = True) -> int:
    """Executes the full RailRouteAgent pipeline: Planner -> Verifier -> Ranker -> Safety Gate.

    Args:
        origin: Origin station code (e.g. "NDLS").
        destination: Destination station code (e.g. "MAO").
        date: Travel date string in YYYY-MM-DD format.
        interactive: Whether to prompt for Human-in-the-Loop safety gate input.

    Returns:
        int: Exit status code (0 for success, 1 for failure/abort).
    """
    console = Console()

    console.print(Panel(
        "[bold cyan][RailRouteAgent] Indian Railways Agentic Split-Journey Planner[/bold cyan]\n"
        "[italic white]Multi-Agent Graph Search, Verification & Risk Audit Engine[/italic white]",
        border_style="cyan"
    ))

    verifier = VerifierAgent()
    ranker = RankingAgent()

    # Step 1 & 2: Planner & Verifier Multi-Agent Reflection Loop
    with console.status("[bold yellow]Planner & Verifier Agents: Generating & auditing split paths...[/bold yellow]", spinner="dots"):
        verified_routes = verifier.verify_and_refine(origin, destination, date)

    if not verified_routes:
        console.print("[bold red]❌ No operationally feasible split-journey routes found matching the mandatory safety criteria.[/bold red]")
        return 1

    # Step 3: Ranking Agent Synthesis
    with console.status("[bold green]Ranking Agent: Synthesizing travel advisory recommendations...[/bold green]", spinner="dots"):
        markdown_report = ranker.rank_and_summarize(verified_routes)

    console.print("\n")
    console.print(Markdown(markdown_report))
    console.print("\n")

    # Step 4: Human-in-the-Loop Safety Gate (Consequential Action Safeguard)
    if interactive:
        try:
            choice = input("⚠️  [ACTION REQUIRED] Select an itinerary number to simulate booking (or press 'Q' to abort): ").strip()
            if choice.upper() == "Q":
                console.print("[yellow]Booking simulation aborted by user.[/yellow]")
                return 0
            elif choice.isdigit() and 1 <= int(choice) <= len(verified_routes):
                selected_idx = int(choice)
                selected_route = verified_routes[selected_idx - 1]
                console.print(Panel(
                    f"[bold green]✅ [SANDBOX] Booking simulation successful for Option {selected_idx}. No actual transaction occurred.[/bold green]\n"
                    f"[bold white]Route ID:[/bold white] {selected_route.get('route_id')}\n"
                    f"[bold white]Junction:[/bold white] {selected_route.get('legs', [{}])[0].get('to_station', 'N/A')}",
                    title="[RailRouteAgent] IRCTC Booking Safety Gate",
                    border_style="green"
                ))
                return 0
            else:
                console.print("[bold red]Invalid selection. Booking simulation cancelled.[/bold red]")
                return 1
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Execution terminated.[/yellow]")
            return 0

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RailRouteAgent: Agentic Split-Journey Planner for Indian Railways"
    )
    parser.add_argument("--origin", type=str, default="NDLS", help="Origin station code (e.g. NDLS, CSMT, BPL)")
    parser.add_argument("--destination", type=str, default="MAO", help="Destination station code (e.g. MAO, CSMT, BPL)")
    parser.add_argument("--date", type=str, default="2026-09-15", help="Travel date in YYYY-MM-DD format")
    parser.add_argument("--non-interactive", action="store_true", help="Run without Human-in-the-Loop prompt")

    args = parser.parse_args()
    status = run_pipeline(
        origin=args.origin,
        destination=args.destination,
        date=args.date,
        interactive=not args.non_interactive
    )
    sys.exit(status)


if __name__ == "__main__":
    main()
