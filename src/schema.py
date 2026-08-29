"""
Core Pydantic v2 schemas for RailRouteAgent.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class Station(BaseModel):
    """Represents a railway station node."""
    code: str = Field(..., description="Unique 3-4 letter station code (e.g. NDLS, CSMT)")
    name: str = Field(..., description="Full station name")
    zone: str = Field(..., description="Railway zone (e.g. NR, CR, WCR, SECR)")
    platforms: int = Field(..., ge=1, description="Number of platforms at station")


class TrainSchedule(BaseModel):
    """Represents a train route schedule."""
    train_no: str = Field(..., description="5-digit train number")
    train_name: str = Field(..., description="Name of the train")
    src_station: str = Field(..., description="Source station code")
    dest_station: str = Field(..., description="Destination station code")
    departure_time: str = Field(..., description="Departure time in HH:MM format")
    arrival_time: str = Field(..., description="Arrival time in HH:MM format")
    day_offset: int = Field(0, ge=0, description="Day offset (0 for same day arrival, 1+ for next days)")
    classes: List[str] = Field(default_factory=list, description="Available class types e.g. ['1A', '2A', '3A', 'SL']")
    avg_delay_mins: int = Field(0, ge=0, description="Average delay in minutes")

    @field_validator("departure_time", "arrival_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate HH:MM time string format."""
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"Time must be in HH:MM format, got '{v}'")
        try:
            hh, mm = int(parts[0]), int(parts[1])
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError(f"Hours must be 0-23 and minutes 0-59 in HH:MM, got '{v}'")
        except ValueError as e:
            raise ValueError(f"Invalid HH:MM time components in '{v}': {e}")
        return f"{hh:02d}:{mm:02d}"


class Leg(BaseModel):
    """Represents an individual leg in a journey."""
    train: TrainSchedule = Field(..., description="Train schedule for this leg")
    from_station: str = Field(..., description="Boarding station code")
    to_station: str = Field(..., description="Alighting station code")
    dep_time: str = Field(..., description="Scheduled departure time (HH:MM)")
    arr_time: str = Field(..., description="Scheduled arrival time (HH:MM)")
    class_type: str = Field(..., description="Travel class (e.g. 2A, 3A, SL)")
    availability_status: str = Field(..., description="IRCTC availability string e.g. AVAILABLE-0012, WL-15")
    confirmation_prob: float = Field(..., ge=0.0, le=1.0, description="Estimated ticket confirmation probability (0.0 to 1.0)")

    @field_validator("dep_time", "arr_time")
    @classmethod
    def validate_leg_time_format(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"Leg time must be in HH:MM format, got '{v}'")
        return v


class SplitItinerary(BaseModel):
    """Represents a complete or split multi-leg itinerary."""
    route_id: str = Field(..., description="Unique route/itinerary identifier")
    origin: str = Field(..., description="Origin station code")
    destination: str = Field(..., description="Destination station code")
    legs: List[Leg] = Field(..., description="List of journey legs making up the itinerary")
    total_duration_mins: int = Field(..., ge=0, description="Total journey time in minutes including layovers")
    layover_buffer_mins: int = Field(0, ge=0, description="Layover buffer between split legs in minutes")
    is_operationally_feasible: bool = Field(..., description="Whether layout/buffer constraints are met")
    feasibility_notes: str = Field("", description="Detailed explanation of operational feasibility")
    overall_confirmation_prob: float = Field(..., ge=0.0, le=1.0, description="Joint confirmation probability across all legs")
