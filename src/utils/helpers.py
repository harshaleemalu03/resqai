# =============================================================
# ResQAI — Helper Utilities
# Shared utility functions across the application
# =============================================================

import re
from datetime import datetime
from typing import Dict, List, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import SEVERITY_LEVELS, MOCK_SHELTERS


def get_severity_config(severity: str) -> Dict:
    """
    Get display configuration for a severity level.

    Args:
        severity: Severity string (CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN)

    Returns:
        Dict with color, emoji, and priority
    """
    return SEVERITY_LEVELS.get(severity.upper(), SEVERITY_LEVELS["UNKNOWN"])


def format_timestamp(dt: datetime = None) -> str:
    """Format a datetime for display."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%H:%M:%S")


def format_date(dt: datetime = None) -> str:
    """Format a date for display."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%B %d, %Y")


def get_nearby_shelters(limit: int = 5) -> List[Dict]:
    """
    Get mock nearby shelter/hospital recommendations.
    In production, this would use geolocation + real DB.

    Args:
        limit: Maximum number of results to return

    Returns:
        List of shelter dicts sorted by distance
    """
    shelters = sorted(MOCK_SHELTERS, key=lambda x: x["distance_km"])
    return shelters[:limit]


def calculate_shelter_availability(shelter: Dict) -> Dict:
    """
    Calculate and return shelter availability information.

    Args:
        shelter: Shelter dict with capacity and occupancy

    Returns:
        Dict with availability stats
    """
    capacity = shelter.get("capacity", 0)
    occupancy = shelter.get("current_occupancy", 0)
    available = max(0, capacity - occupancy)
    pct = (occupancy / capacity * 100) if capacity > 0 else 0

    if pct >= 95:
        status = "FULL"
        status_color = "#FF2D2D"
    elif pct >= 75:
        status = "ALMOST FULL"
        status_color = "#FF6B00"
    elif pct >= 50:
        status = "MODERATE"
        status_color = "#FFD700"
    else:
        status = "AVAILABLE"
        status_color = "#00C851"

    return {
        "available_spots": available,
        "occupancy_pct": round(pct, 1),
        "status": status,
        "status_color": status_color,
    }


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3].rsplit(' ', 1)[0] + "..."


def sanitize_input(text: str) -> str:
    """Basic sanitization of user input text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Limit length
    return text[:2000]


def build_emergency_kit_checklist() -> Dict[str, List[str]]:
    """Return a standard emergency kit checklist by category."""
    return {
        "Water & Food": [
            "Water (1 gallon/person/day for 3 days)",
            "Non-perishable food (3-day supply)",
            "Manual can opener",
            "Water purification tablets",
        ],
        "Medical": [
            "First aid kit",
            "Prescription medications (7-day supply)",
            "Extra glasses or contact lenses",
            "Medical records copies",
        ],
        "Communication": [
            "Battery-powered or hand-crank radio",
            "Flashlights with extra batteries",
            "Whistle (to signal for help)",
            "Cell phone with chargers and backup battery",
        ],
        "Documents": [
            "Copies of important documents",
            "Cash in small bills",
            "Emergency contact list",
            "Local maps",
        ],
        "Clothing & Shelter": [
            "Warm clothing for each family member",
            "Rain gear",
            "Sturdy shoes",
            "Sleeping bags or blankets",
        ],
        "Tools": [
            "Wrench/pliers to shut off utilities",
            "Dust masks",
            "Plastic sheeting and duct tape",
            "Moist towelettes and garbage bags",
        ],
    }
