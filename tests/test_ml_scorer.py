"""
Unit tests for RailRouteAgent ML PNR confirmation probability scorer tool.
"""

import pytest
from src.tools import predict_pnr_confirmation


def test_available_status():
    assert predict_pnr_confirmation("AVAILABLE-0012", "3A", 5) == 1.0
    assert predict_pnr_confirmation("AVAILABLE", "2A", 1) == 1.0


def test_regret_status():
    assert predict_pnr_confirmation("REGRET", "SL", 3) == 0.0
    assert predict_pnr_confirmation("REGRET-0000", "3A", 10) == 0.0


def test_rac_status():
    prob_rac1 = predict_pnr_confirmation("RAC-05", "3A", 7)
    prob_rac2 = predict_pnr_confirmation("RAC-25", "3A", 1)
    assert 0.80 <= prob_rac1 <= 0.98
    assert 0.80 <= prob_rac2 <= 0.98
    assert prob_rac1 > prob_rac2  # Lower RAC number has higher probability


def test_waitlist_exponential_decay():
    prob_wl5 = predict_pnr_confirmation("GNWL-5", "3A", 5)
    prob_wl30 = predict_pnr_confirmation("GNWL-30", "3A", 5)
    prob_wl80 = predict_pnr_confirmation("GNWL-80", "3A", 5)

    assert prob_wl5 > prob_wl30 > prob_wl80
    assert 0.01 <= prob_wl80 <= 0.99


def test_quota_differences():
    # GNWL moves faster than PQWL
    prob_gnwl = predict_pnr_confirmation("GNWL-20", "3A", 5)
    prob_pqwl = predict_pnr_confirmation("PQWL-20", "3A", 5)
    assert prob_gnwl > prob_pqwl


def test_travel_class_multipliers():
    # Sleeper (SL) has higher cancellation volume than 1A / 2A
    prob_sl = predict_pnr_confirmation("GNWL-15", "SL", 5)
    prob_1a = predict_pnr_confirmation("GNWL-15", "1A", 5)
    assert prob_sl > prob_1a


test_travel_class_multipliers()


def test_days_to_journey_movement():
    # More days remaining = higher probability of confirmation movement
    prob_10days = predict_pnr_confirmation("GNWL-15", "3A", 10)
    prob_1day = predict_pnr_confirmation("GNWL-15", "3A", 1)
    assert prob_10days > prob_1day


def test_malformed_inputs():
    # Malformed waitlist string without numbers
    prob_malformed = predict_pnr_confirmation("WL", "3A", 3)
    assert 0.01 <= prob_malformed <= 0.99

    # Unknown travel class
    prob_unknown_class = predict_pnr_confirmation("GNWL-10", "UNKNOWN", 3)
    assert 0.01 <= prob_unknown_class <= 0.99

    # Negative days
    prob_neg_days = predict_pnr_confirmation("GNWL-10", "3A", -5)
    assert 0.01 <= prob_neg_days <= 0.99
