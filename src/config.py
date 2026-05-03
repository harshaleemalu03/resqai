# =============================================================
# ResQAI — Configuration Module
# Central configuration for all settings and constants
# =============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ─── RAG Settings ─────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_RETRIEVAL_DOCS = 5
EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model

# ─── LLM Settings ─────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"
MAX_TOKENS = 2048
TEMPERATURE = 0.3  # Lower for more focused emergency responses

# ─── App Settings ─────────────────────────────────────────
APP_TITLE = "ResQAI"
APP_SUBTITLE = "AI Disaster Response & Relief Coordinator"
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", 10))
MAX_CHAT_HISTORY = 20

# ─── Severity Levels ──────────────────────────────────────
SEVERITY_LEVELS = {
    "CRITICAL": {"color": "#FF2D2D", "emoji": "🔴", "priority": 1},
    "HIGH":     {"color": "#FF6B00", "emoji": "🟠", "priority": 2},
    "MEDIUM":   {"color": "#FFD700", "emoji": "🟡", "priority": 3},
    "LOW":      {"color": "#00C851", "emoji": "🟢", "priority": 4},
    "UNKNOWN":  {"color": "#888888", "emoji": "⚪", "priority": 5},
}

# ─── Disaster Types ───────────────────────────────────────
DISASTER_TYPES = [
    "Earthquake", "Flood", "Hurricane/Cyclone", "Tornado",
    "Wildfire", "Tsunami", "Landslide", "Drought",
    "Chemical Spill", "Nuclear Incident", "Pandemic",
    "Building Collapse", "Avalanche", "Volcanic Eruption",
    "General Emergency", "Unknown"
]

# ─── Mock Shelter Data ────────────────────────────────────
MOCK_SHELTERS = [
    {
        "name": "City Emergency Shelter - Central",
        "type": "Emergency Shelter",
        "address": "123 Main St, City Center",
        "capacity": 500,
        "current_occupancy": 210,
        "contact": "+1-800-555-0101",
        "services": ["Food", "Water", "Medical", "Beds"],
        "distance_km": 1.2,
    },
    {
        "name": "Community Relief Center - North",
        "type": "Relief Center",
        "address": "45 North Ave, Uptown",
        "capacity": 300,
        "current_occupancy": 89,
        "contact": "+1-800-555-0102",
        "services": ["Food", "Water", "Clothing"],
        "distance_km": 2.8,
    },
    {
        "name": "Regional Medical Camp",
        "type": "Medical Facility",
        "address": "78 Hospital Rd, East District",
        "capacity": 150,
        "current_occupancy": 67,
        "contact": "+1-800-555-0103",
        "services": ["Emergency Medical", "Surgery", "ICU"],
        "distance_km": 3.5,
    },
    {
        "name": "School Relief Hub - South",
        "type": "Emergency Shelter",
        "address": "200 Education Blvd, South Side",
        "capacity": 400,
        "current_occupancy": 312,
        "contact": "+1-800-555-0104",
        "services": ["Food", "Water", "Beds", "Child Care"],
        "distance_km": 4.1,
    },
    {
        "name": "Sports Arena Emergency Camp",
        "type": "Large Shelter",
        "address": "1 Arena Drive, West End",
        "capacity": 2000,
        "current_occupancy": 445,
        "contact": "+1-800-555-0105",
        "services": ["Food", "Water", "Medical", "Beds", "Pet Care"],
        "distance_km": 5.6,
    },
]

# ─── Emergency Hotlines ───────────────────────────────────
EMERGENCY_HOTLINES = {
    "Emergency Services": "911",
    "FEMA Helpline": "1-800-621-3362",
    "Disaster Distress Helpline": "1-800-985-5990",
    "Red Cross": "1-800-733-2767",
    "National Crisis Line": "988",
    "Poison Control": "1-800-222-1222",
    "Coast Guard": "1-800-368-5647",
}
