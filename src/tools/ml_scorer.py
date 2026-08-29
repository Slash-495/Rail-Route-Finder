"""
PNR Confirmation ML Scorer Tool for RailRouteAgent.
Provides a lightweight, sub-millisecond Quantile Regression mathematical model
to predict the probability of a waitlisted Indian Railways ticket getting confirmed.
"""

import math
import re
from typing import Dict

# Quota movement factor multiplier
QUOTA_FACTORS: Dict[str, float] = {
    "GNWL": 1.00,  # General Waitlist (highest cancellation movement)
    "RLWL": 0.85,  # Remote Location Waitlist (moderate movement)
    "PQWL": 0.70,  # Pooled Quota Waitlist (lower movement)
    "RSWL": 0.75,  # Roadside Station Waitlist
    "WL": 0.90,    # Generic Waitlist fallback
}

# Class cancellation volume multiplier
CLASS_MULTIPLIERS: Dict[str, float] = {
    "SL": 1.20,   # Sleeper Class (highest cancellation volume -> higher prob)
    "2S": 1.10,   # Second Seating
    "CC": 1.05,   # Chair Car
    "3A": 1.00,   # 3-Tier AC (baseline)
    "3E": 1.00,   # 3-Tier AC Economy
    "2A": 0.85,   # 2-Tier AC
    "1A": 0.70,   # 1-Tier AC (lowest cancellation volume -> lower prob)
}


def predict_pnr_confirmation(current_status: str, travel_class: str, days_to_journey: int) -> float:
    """Predicts the probability of an IRCTC ticket getting confirmed (0.0 to 1.0).

    Simulates a Quantile Regression model using exponential decay curves, quota factors,
    cancellation volume multipliers by travel class, and logarithmic time-to-journey scaling.

    Args:
        current_status: Current IRCTC status string (e.g. "AVAILABLE-0012", "RAC-14", "WL-5", "GNWL-45", "PQWL-10", "REGRET").
        travel_class: Travel class code (e.g. "1A", "2A", "3A", "SL", "CC", "2S").
        days_to_journey: Number of days remaining before departure.

    Returns:
        float: Confirmation probability score rounded to 4 decimal places (clamped between 0.01 and 0.99 for waitlists,
        1.0 for AVAILABLE, 0.0 for REGRET).
    """
    if not current_status:
        return 0.50

    status_upper = current_status.upper().strip()
    class_upper = travel_class.upper().strip() if travel_class else "3A"
    safe_days = max(0, days_to_journey)

    # 1. AVAILABLE status
    if "AVAILABLE" in status_upper:
        return 1.0

    # 2. REGRET status
    if "REGRET" in status_upper:
        return 0.0

    # 3. RAC (Reservation Against Cancellation) status
    if "RAC" in status_upper:
        # Extract RAC number if available (e.g. RAC-14 -> 14)
        rac_match = re.search(r"\d+", status_upper)
        rac_no = int(rac_match.group()) if rac_match else 10
        
        # RAC has very high confirmation probability (0.80 to 0.98)
        base_rac_prob = 0.95 - (min(rac_no, 30) * 0.005) + (min(safe_days, 15) * 0.003)
        return round(min(0.98, max(0.80, base_rac_prob)), 4)

    # 4. Waitlist status ("WL", "GNWL", "PQWL", "RLWL")
    # Extract waitlist position number
    number_match = re.search(r"\d+", status_upper)
    if number_match:
        wl_number = int(number_match.group())
    else:
        wl_number = 20  # Default fallback if string is malformed e.g. "WL"

    # Identify quota type prefix
    quota_factor = 0.90
    for q_prefix, q_val in QUOTA_FACTORS.items():
        if q_prefix in status_upper:
            quota_factor = q_val
            break

    # Get class cancellation volume multiplier
    class_mult = CLASS_MULTIPLIERS.get(class_upper, 1.00)

    # Days to journey multiplier: more days left = logarithmic boost for cancellation movement
    days_mult = 1.0 + 0.05 * math.log(1.0 + safe_days)

    # Exponential decay curve: base_prob = exp(-lambda * wl_number)
    # Lambda rate scales inversely with quota factor
    lambda_decay = 0.035 / quota_factor
    base_prob = math.exp(-lambda_decay * wl_number)

    # Combined probability
    raw_prob = base_prob * class_mult * days_mult

    # Clamp result strictly between 0.01 and 0.99 for waitlists
    clamped_prob = min(0.99, max(0.01, raw_prob))
    return round(clamped_prob, 4)
