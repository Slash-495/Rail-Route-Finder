"""
Configuration settings and directory paths for RailRouteAgent.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Main timetable dataset path
TRAIN_NETWORK_JSON = PROCESSED_DATA_DIR / "train_network.json"
TRAIN_NETWORK_CSV = PROCESSED_DATA_DIR / "train_network.csv"

# Benchmarks directory
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
TEST_CASES_JSON = BENCHMARKS_DIR / "test_cases.json"

# Logs directory
LOGS_DIR = BASE_DIR / "logs"
TRAJECTORIES_DIR = LOGS_DIR / "trajectories"
BENCHMARK_RESULTS_JSON = LOGS_DIR / "benchmark_results.json"

# API keys and parameters
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Operational feasibility thresholds
MIN_LAYOVER_BUFFER_MINS = 30
MAX_LAYOVER_BUFFER_MINS = 360
DEFAULT_CLASS_PRIORITY = ["1A", "2A", "3A", "SL", "CC", "2S"]

# Default HTTP headers for web/API queries in case Streamlit Cloud IP is rate-limited
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

