"""
Synthetic Indian Railways Timetable Data Generator for RailRouteAgent.
Generates realistic network dataset saved to data/processed/train_network.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import PROCESSED_DATA_DIR, TRAIN_NETWORK_JSON
from src.schema import Station, TrainSchedule, Leg


# Key Indian Railways Trunk Stations
SAMPLE_STATIONS: List[Dict[str, Any]] = [
    {"code": "NDLS", "name": "New Delhi", "zone": "NR", "platforms": 16},
    {"code": "CSMT", "name": "Mumbai CSMT", "zone": "CR", "platforms": 18},
    {"code": "BPL", "name": "Bhopal Junction", "zone": "WCR", "platforms": 6},
    {"code": "NGP", "name": "Nagpur Junction", "zone": "CR", "platforms": 8},
    {"code": "JBP", "name": "Jabalpur Junction", "zone": "WCR", "platforms": 6},
    {"code": "MAO", "name": "Madgaon Junction", "zone": "KR", "platforms": 4},
    {"code": "PUNE", "name": "Pune Junction", "zone": "CR", "platforms": 6},
    {"code": "CNB", "name": "Kanpur Central", "zone": "NCR", "platforms": 10},
    {"code": "PRYJ", "name": "Prayagraj Junction", "zone": "NCR", "platforms": 10},
    {"code": "BSB", "name": "Varanasi Junction", "zone": "NR", "platforms": 9},
    {"code": "HWH", "name": "Howrah Junction", "zone": "ER", "platforms": 23},
    {"code": "ADI", "name": "Ahmedabad Junction", "zone": "WR", "platforms": 12},
]

# Connected Train Schedules with realistic IRCTC availability and delays
SAMPLE_TRAINS: List[Dict[str, Any]] = [
    # NDLS routes
    {
        "train_no": "12952", "train_name": "New Delhi Rajdhani", "src_station": "NDLS", "dest_station": "CSMT",
        "departure_time": "16:55", "arrival_time": "08:35", "day_offset": 1, "classes": ["1A", "2A", "3A"],
        "avg_delay_mins": 15, "availability_status": "AVAILABLE-0012", "confirmation_prob": 0.95
    },
    {
        "train_no": "12138", "train_name": "Punjab Mail", "src_station": "NDLS", "dest_station": "CSMT",
        "departure_time": "05:15", "arrival_time": "07:35", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 45, "availability_status": "WL-15", "confirmation_prob": 0.45
    },
    {
        "train_no": "12002", "train_name": "Bhopal Shatabdi", "src_station": "NDLS", "dest_station": "BPL",
        "departure_time": "06:00", "arrival_time": "14:05", "day_offset": 0, "classes": ["EC", "CC"],
        "avg_delay_mins": 10, "availability_status": "AVAILABLE-0042", "confirmation_prob": 0.98
    },
    {
        "train_no": "12628", "train_name": "Karnataka Express", "src_station": "NDLS", "dest_station": "BPL",
        "departure_time": "20:20", "arrival_time": "04:30", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 25, "availability_status": "GNWL-42", "confirmation_prob": 0.30
    },
    {
        "train_no": "12622", "train_name": "Tamil Nadu Express", "src_station": "NDLS", "dest_station": "NGP",
        "departure_time": "22:30", "arrival_time": "13:05", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 30, "availability_status": "AVAILABLE-0005", "confirmation_prob": 0.90
    },
    {
        "train_no": "12410", "train_name": "Gondwana Express", "src_station": "NDLS", "dest_station": "JBP",
        "departure_time": "15:05", "arrival_time": "03:00", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 20, "availability_status": "REGRET", "confirmation_prob": 0.05
    },
    {
        "train_no": "12802", "train_name": "Purushottam Express", "src_station": "NDLS", "dest_station": "CNB",
        "departure_time": "22:40", "arrival_time": "04:00", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 35, "availability_status": "AVAILABLE-0088", "confirmation_prob": 0.99
    },
    {
        "train_no": "12424", "train_name": "Dibrugarh Rajdhani", "src_station": "NDLS", "dest_station": "PRYJ",
        "departure_time": "16:20", "arrival_time": "23:10", "day_offset": 0, "classes": ["1A", "2A", "3A"],
        "avg_delay_mins": 15, "availability_status": "AVAILABLE-0018", "confirmation_prob": 0.94
    },

    # BPL routes
    {
        "train_no": "12156", "train_name": "Shaan-e-Bhopal Express", "src_station": "BPL", "dest_station": "NDLS",
        "departure_time": "23:05", "arrival_time": "06:20", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 10, "availability_status": "AVAILABLE-0030", "confirmation_prob": 0.96
    },
    {
        "train_no": "12160", "train_name": "Jabalpur Express", "src_station": "BPL", "dest_station": "JBP",
        "departure_time": "22:00", "arrival_time": "03:30", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 15, "availability_status": "RAC-08", "confirmation_prob": 0.75
    },
    {
        "train_no": "12061", "train_name": "Jan Shatabdi Express", "src_station": "BPL", "dest_station": "JBP",
        "departure_time": "17:40", "arrival_time": "22:55", "day_offset": 0, "classes": ["CC", "2S"],
        "avg_delay_mins": 10, "availability_status": "AVAILABLE-0120", "confirmation_prob": 0.99
    },
    {
        "train_no": "12616", "train_name": "Grand Trunk Express", "src_station": "BPL", "dest_station": "NGP",
        "departure_time": "03:20", "arrival_time": "08:35", "day_offset": 0, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 40, "availability_status": "WL-28", "confirmation_prob": 0.35
    },
    {
        "train_no": "12137", "train_name": "Punjab Mail (Down)", "src_station": "BPL", "dest_station": "CSMT",
        "departure_time": "09:55", "arrival_time": "19:35", "day_offset": 0, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 30, "availability_status": "AVAILABLE-0015", "confirmation_prob": 0.91
    },

    # NGP routes
    {
        "train_no": "12290", "train_name": "Nagpur Duronto", "src_station": "NGP", "dest_station": "CSMT",
        "departure_time": "20:40", "arrival_time": "08:05", "day_offset": 1, "classes": ["1A", "2A", "3A"],
        "avg_delay_mins": 20, "availability_status": "AVAILABLE-0004", "confirmation_prob": 0.88
    },
    {
        "train_no": "12106", "train_name": "Vidarbha Express", "src_station": "NGP", "dest_station": "CSMT",
        "departure_time": "17:00", "arrival_time": "06:15", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 25, "availability_status": "WL-08", "confirmation_prob": 0.60
    },
    {
        "train_no": "12114", "train_name": "Pune Garib Rath", "src_station": "NGP", "dest_station": "PUNE",
        "departure_time": "18:00", "arrival_time": "06:25", "day_offset": 1, "classes": ["3A"],
        "avg_delay_mins": 15, "availability_status": "AVAILABLE-0064", "confirmation_prob": 0.97
    },
    {
        "train_no": "12834", "train_name": "Howrah Express", "src_station": "NGP", "dest_station": "HWH",
        "departure_time": "19:00", "arrival_time": "13:25", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 50, "availability_status": "GNWL-19", "confirmation_prob": 0.40
    },

    # JBP routes
    {
        "train_no": "12187", "train_name": "Jabalpur Garib Rath", "src_station": "JBP", "dest_station": "CSMT",
        "departure_time": "19:50", "arrival_time": "12:20", "day_offset": 1, "classes": ["3A"],
        "avg_delay_mins": 30, "availability_status": "AVAILABLE-0022", "confirmation_prob": 0.93
    },
    {
        "train_no": "11072", "train_name": "Kamayani Express", "src_station": "JBP", "dest_station": "BSB",
        "departure_time": "15:30", "arrival_time": "23:45", "day_offset": 0, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 35, "availability_status": "WL-12", "confirmation_prob": 0.50
    },

    # CSMT & MAO & PUNE routes
    {
        "train_no": "12124", "train_name": "Deccan Queen", "src_station": "CSMT", "dest_station": "PUNE",
        "departure_time": "17:10", "arrival_time": "20:25", "day_offset": 0, "classes": ["CC", "2S"],
        "avg_delay_mins": 5, "availability_status": "AVAILABLE-0150", "confirmation_prob": 0.99
    },
    {
        "train_no": "11139", "train_name": "CSMT Gadag Express", "src_station": "CSMT", "dest_station": "MAO",
        "departure_time": "21:20", "arrival_time": "08:30", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 25, "availability_status": "AVAILABLE-0010", "confirmation_prob": 0.92
    },
    {
        "train_no": "10111", "train_name": "Konkan Kanya Express", "src_station": "CSMT", "dest_station": "MAO",
        "departure_time": "23:00", "arrival_time": "09:45", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 20, "availability_status": "WL-34", "confirmation_prob": 0.32
    },
    {
        "train_no": "12134", "train_name": "Mangaluru Express", "src_station": "MAO", "dest_station": "CSMT",
        "departure_time": "19:05", "arrival_time": "04:35", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 15, "availability_status": "AVAILABLE-0008", "confirmation_prob": 0.89
    },
    {
        "train_no": "12780", "train_name": "Goa Express", "src_station": "PUNE", "dest_station": "MAO",
        "departure_time": "16:35", "arrival_time": "05:40", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 20, "availability_status": "AVAILABLE-0025", "confirmation_prob": 0.95
    },
    {
        "train_no": "12130", "train_name": "Azad Hind Express", "src_station": "PUNE", "dest_station": "NGP",
        "departure_time": "18:35", "arrival_time": "09:15", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 40, "availability_status": "WL-52", "confirmation_prob": 0.25
    },
    {
        "train_no": "12779", "train_name": "Goa Express (Return)", "src_station": "MAO", "dest_station": "PUNE",
        "departure_time": "15:00", "arrival_time": "03:55", "day_offset": 1, "classes": ["2A", "3A", "SL"],
        "avg_delay_mins": 15, "availability_status": "AVAILABLE-0018", "confirmation_prob": 0.94
    },

    # Eastern trunk routes (CNB / PRYJ / HWH / BSB)
    {
        "train_no": "12304", "train_name": "Poorva Express", "src_station": "CNB", "dest_station": "HWH",
        "departure_time": "23:05", "arrival_time": "17:00", "day_offset": 1, "classes": ["1A", "2A", "3A", "SL"],
        "avg_delay_mins": 60, "availability_status": "REGRET", "confirmation_prob": 0.00
    },
    {
        "train_no": "12274", "train_name": "Howrah Duronto", "src_station": "PRYJ", "dest_station": "HWH",
        "departure_time": "01:15", "arrival_time": "12:40", "day_offset": 0, "classes": ["1A", "2A", "3A"],
        "avg_delay_mins": 20, "availability_status": "AVAILABLE-0014", "confirmation_prob": 0.92
    },
    {
        "train_no": "12932", "train_name": "Ahmedabad Double Decker", "src_station": "ADI", "dest_station": "CSMT",
        "departure_time": "06:00", "arrival_time": "13:00", "day_offset": 0, "classes": ["CC"],
        "avg_delay_mins": 10, "availability_status": "AVAILABLE-0095", "confirmation_prob": 0.98
    },
]


def generate_sample_dataset() -> Dict[str, Any]:
    """Validate and build the sample timetable network dictionary."""

    # Validate stations with Pydantic model
    validated_stations = [Station(**s).model_dump() for s in SAMPLE_STATIONS]

    # Validate train schedules with Pydantic model
    validated_trains = []
    for t in SAMPLE_TRAINS:
        train_data = {
            "train_no": t["train_no"],
            "train_name": t["train_name"],
            "src_station": t["src_station"],
            "dest_station": t["dest_station"],
            "departure_time": t["departure_time"],
            "arrival_time": t["arrival_time"],
            "day_offset": t["day_offset"],
            "classes": t["classes"],
            "avg_delay_mins": t["avg_delay_mins"]
        }
        # Run through Pydantic validator
        ts = TrainSchedule(**train_data)
        
        # Include leg-specific live attributes
        validated_trains.append({
            **ts.model_dump(),
            "availability_status": t["availability_status"],
            "confirmation_prob": t["confirmation_prob"]
        })

    # Construct NetworkX graph to compute connectivity
    G = nx.MultiDiGraph()
    for s in validated_stations:
        G.add_node(s["code"], **s)

    for tr in validated_trains:
        G.add_edge(tr["src_station"], tr["dest_station"], key=tr["train_no"], **tr)

    network_data = {
        "metadata": {
            "version": "1.0.0",
            "description": "Synthetic Indian Railways timetable network for RailRouteAgent",
            "station_count": len(validated_stations),
            "train_route_count": len(validated_trains),
            "is_connected": nx.is_weakly_connected(G),
        },
        "stations": validated_stations,
        "trains": validated_trains
    }

    return network_data


def save_dataset(output_path: Path = TRAIN_NETWORK_JSON) -> Dict[str, Any]:
    """Generate and write train network JSON dataset to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_sample_dataset()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return dataset


def display_cli_summary(dataset: Dict[str, Any], output_path: Path):
    """Render beautiful summary output using rich."""
    console = Console()

    table = Table(title="[RailRouteAgent] Synthetic Timetable Dataset Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=40)

    table.add_row("Total Station Nodes", str(dataset["metadata"]["station_count"]))
    table.add_row("Total Connected Train Edges", str(dataset["metadata"]["train_route_count"]))
    table.add_row("Network Weakly Connected", str(dataset["metadata"]["is_connected"]))
    table.add_row("Dataset File Path", str(output_path.resolve()))

    console.print(Panel(table, border_style="bright_blue", title="Dataset Generation Successful"))

    # Print Station Breakdown Table
    station_table = Table(title="Included Station Nodes", show_header=True, header_style="bold yellow")
    station_table.add_column("Code", style="bold green", width=10)
    station_table.add_column("Name", width=25)
    station_table.add_column("Zone", width=10)
    station_table.add_column("Platforms", justify="right", width=10)

    for st in dataset["stations"]:
        station_table.add_row(st["code"], st["name"], st["zone"], str(st["platforms"]))

    console.print(station_table)


if __name__ == "__main__":
    ds = save_dataset(TRAIN_NETWORK_JSON)
    display_cli_summary(ds, TRAIN_NETWORK_JSON)
