# =============================================================
# ResQAI — Main Streamlit Application
# AI Disaster Response & Relief Coordinator
# =============================================================

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

# ─── Path Setup ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import (
    APP_TITLE, APP_SUBTITLE, SEVERITY_LEVELS,
    EMERGENCY_HOTLINES, MAX_CHAT_HISTORY, GEMINI_API_KEY,
    OPENWEATHER_API_KEY
)
from src.rag.pipeline import (
    FAISSVectorStore, extract_text_from_pdf,
    chunk_text, build_rag_context, compute_file_hash
)
from src.agents.responder import RAGChatAgent, SeverityClassifier
from src.utils.weather import fetch_weather, get_weather_risk_level
from src.utils.helpers import (
    get_nearby_shelters, calculate_shelter_availability,
    format_timestamp, format_date, get_severity_config,
    build_emergency_kit_checklist
)

# =============================================================
# PAGE CONFIG — Must be first Streamlit call
# =============================================================
st.set_page_config(
    page_title="ResQAI — Disaster Response",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# CUSTOM CSS — Dark Military/Emergency Theme
# =============================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0a0c0f;
    --bg-secondary:  #111418;
    --bg-card:       #161b22;
    --bg-card-hover: #1c2333;
    --border-color:  #21262d;
    --border-accent: #30363d;
    --red-critical:  #ff2d2d;
    --orange-high:   #ff6b00;
    --yellow-medium: #ffd700;
    --green-low:     #00c851;
    --blue-info:     #2196f3;
    --teal-accent:   #00bcd4;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
    --font-display:  'Bebas Neue', 'Impact', sans-serif;
    --font-mono:     'Share Tech Mono', 'Courier New', monospace;
    --font-body:     'Rajdhani', 'Arial', sans-serif;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 14px;
    font-weight: 500;
}

/* ── Main Header ── */
.resqai-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid var(--red-critical);
    border-radius: 4px;
    padding: 20px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.resqai-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--red-critical), transparent);
}
.resqai-header-title {
    font-family: var(--font-display) !important;
    font-size: 42px;
    letter-spacing: 4px;
    color: var(--red-critical) !important;
    margin: 0;
    line-height: 1;
    text-shadow: 0 0 20px rgba(255,45,45,0.5);
}
.resqai-header-subtitle {
    font-family: var(--font-mono) !important;
    font-size: 12px;
    color: var(--teal-accent);
    letter-spacing: 2px;
    margin-top: 4px;
    text-transform: uppercase;
}
.resqai-header-status {
    font-family: var(--font-mono) !important;
    font-size: 11px;
    color: var(--green-low);
    text-align: right;
}

/* ── Severity Badges ── */
.severity-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: var(--font-mono) !important;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.severity-critical { background: rgba(255,45,45,0.2); color: #ff2d2d; border: 1px solid #ff2d2d; }
.severity-high     { background: rgba(255,107,0,0.2); color: #ff6b00; border: 1px solid #ff6b00; }
.severity-medium   { background: rgba(255,215,0,0.2); color: #ffd700; border: 1px solid #ffd700; }
.severity-low      { background: rgba(0,200,81,0.2);  color: #00c851; border: 1px solid #00c851; }
.severity-unknown  { background: rgba(136,136,136,0.2); color: #888; border: 1px solid #888; }

/* ── Cards ── */
.resq-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.resq-card:hover { border-color: var(--border-accent); }
.resq-card-title {
    font-family: var(--font-mono) !important;
    font-size: 11px;
    color: var(--text-secondary);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.resq-card-value {
    font-family: var(--font-display) !important;
    font-size: 28px;
    color: var(--text-primary);
    line-height: 1;
}

/* ── Alert Box ── */
.alert-critical {
    background: rgba(255,45,45,0.08);
    border-left: 3px solid var(--red-critical);
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: var(--font-body) !important;
    font-size: 14px;
    color: #ffb3b3;
}
.alert-info {
    background: rgba(33,150,243,0.08);
    border-left: 3px solid var(--blue-info);
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: var(--font-body) !important;
    font-size: 14px;
    color: #90caf9;
}
.alert-success {
    background: rgba(0,200,81,0.08);
    border-left: 3px solid var(--green-low);
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: var(--font-body) !important;
    font-size: 14px;
    color: #a5d6a7;
}
.alert-warning {
    background: rgba(255,215,0,0.08);
    border-left: 3px solid var(--yellow-medium);
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: var(--font-body) !important;
    font-size: 14px;
    color: #fff59d;
}

/* ── Chat Messages ── */
.chat-message {
    display: flex;
    gap: 12px;
    margin: 10px 0;
    padding: 12px;
    border-radius: 4px;
    font-family: var(--font-body) !important;
    font-size: 15px;
    line-height: 1.5;
}
.chat-user {
    background: rgba(33,150,243,0.08);
    border: 1px solid rgba(33,150,243,0.2);
    flex-direction: row-reverse;
}
.chat-assistant {
    background: rgba(0,188,212,0.06);
    border: 1px solid rgba(0,188,212,0.15);
}
.chat-avatar {
    width: 32px;
    height: 32px;
    border-radius: 2px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.chat-avatar-user      { background: rgba(33,150,243,0.3); }
.chat-avatar-assistant { background: rgba(255,45,45,0.3); }
.chat-content { flex: 1; color: var(--text-primary); }
.chat-meta {
    font-family: var(--font-mono) !important;
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 4px;
}

/* ── Shelter Card ── */
.shelter-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 10px;
}
.shelter-name {
    font-family: var(--font-body) !important;
    font-weight: 700;
    font-size: 15px;
    color: var(--teal-accent);
    margin-bottom: 4px;
}
.shelter-type {
    font-family: var(--font-mono) !important;
    font-size: 10px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.shelter-stat {
    font-family: var(--font-mono) !important;
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
}
.shelter-bar {
    background: var(--border-color);
    height: 4px;
    border-radius: 2px;
    margin: 6px 0;
    overflow: hidden;
}
.shelter-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}

/* ── Weather Widget ── */
.weather-widget {
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 16px;
    text-align: center;
}
.weather-temp {
    font-family: var(--font-display) !important;
    font-size: 48px;
    color: var(--text-primary);
    line-height: 1;
}
.weather-desc {
    font-family: var(--font-body) !important;
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: 4px;
}
.weather-detail {
    font-family: var(--font-mono) !important;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Agent Thought Steps ── */
.thought-step {
    background: rgba(0,188,212,0.04);
    border: 1px solid rgba(0,188,212,0.15);
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: var(--font-mono) !important;
    font-size: 12px;
}
.thought-step-title {
    color: var(--teal-accent);
    font-weight: bold;
    margin-bottom: 4px;
}
.thought-step-content { color: var(--text-secondary); }

/* ── Progress Bar Override ── */
.stProgress > div > div { background-color: var(--teal-accent) !important; }

/* ── Button Overrides ── */
.stButton > button {
    background: rgba(255,45,45,0.1) !important;
    border: 1px solid rgba(255,45,45,0.4) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(255,45,45,0.2) !important;
    border-color: var(--red-critical) !important;
}

/* ── Input Fields ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    border-radius: 2px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--teal-accent) !important;
    box-shadow: 0 0 0 2px rgba(0,188,212,0.1) !important;
}

/* ── File Uploader ── */
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: 4px !important;
    color: var(--text-secondary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--teal-accent) !important;
    border-bottom: 2px solid var(--teal-accent) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    border-radius: 2px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 12px;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 28px !important;
    color: var(--text-primary) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-color) !important;
    margin: 16px 0;
}

/* ── Hotlines Table ── */
.hotline-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-color);
    font-family: var(--font-body) !important;
    font-size: 13px;
}
.hotline-item:last-child { border-bottom: none; }
.hotline-name { color: var(--text-secondary); }
.hotline-number {
    font-family: var(--font-mono) !important;
    color: var(--teal-accent);
    font-weight: bold;
}

/* ── Section Headers ── */
.section-header {
    font-family: var(--font-mono) !important;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
    margin-bottom: 12px;
}

/* ── Upload Success ── */
.doc-tag {
    display: inline-block;
    background: rgba(0,200,81,0.1);
    border: 1px solid rgba(0,200,81,0.3);
    border-radius: 2px;
    padding: 2px 8px;
    margin: 2px;
    font-family: var(--font-mono) !important;
    font-size: 11px;
    color: var(--green-low);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# =============================================================
# SESSION STATE INITIALIZATION
# =============================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "chat_history": [],
        "vector_store": FAISSVectorStore(),
        "uploaded_docs": [],
        "weather_data": None,
        "weather_city": "New York",
        "current_severity": "UNKNOWN",
        "current_disaster": "Unknown",
        "agent": RAGChatAgent(),
        "severity_classifier": SeverityClassifier(),
        "processing_pdf": False,
        "total_chunks": 0,
        "last_analysis": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# =============================================================
# SIDEBAR
# =============================================================
def render_sidebar():
    with st.sidebar:
        # ── Logo ────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px;">
            <div style="font-family:'Bebas Neue',sans-serif; font-size:32px;
                        color:#ff2d2d; letter-spacing:4px;
                        text-shadow: 0 0 15px rgba(255,45,45,0.5);">
                🆘 RESQAI
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                        color:#00bcd4; letter-spacing:2px; margin-top:2px;">
                DISASTER RESPONSE SYSTEM
            </div>
        </div>
        <hr style="border-color:#21262d; margin:8px 0 16px;">
        """, unsafe_allow_html=True)

        # ── Navigation ──────────────────────────────────────
        st.markdown('<div class="section-header">Navigation</div>',
                    unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            ["🏠 Dashboard", "💬 Emergency Chat", "📚 Document Upload",
             "🏥 Find Shelter", "🌤 Weather Monitor", "📋 Emergency Kit"],
            label_visibility="collapsed"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Live Status ─────────────────────────────────────
        st.markdown('<div class="section-header">System Status</div>',
                    unsafe_allow_html=True)

        docs_count = len(st.session_state.uploaded_docs)
        chunks = st.session_state.total_chunks

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", docs_count, label_visibility="visible")
        with col2:
            st.metric("Chunks", chunks, label_visibility="visible")

        api_ok = bool(GEMINI_API_KEY)
        weather_ok = bool(OPENWEATHER_API_KEY)
        st.markdown(f"""
        <div style="margin-top:8px; font-family:'Share Tech Mono',monospace; font-size:11px;">
            <div>{'🟢' if api_ok else '🔴'} Gemini AI {'Online' if api_ok else 'No Key'}</div>
            <div style="margin-top:4px;">{'🟢' if weather_ok else '🟡'} Weather API {'Online' if weather_ok else 'Demo Mode'}</div>
            <div style="margin-top:4px;">🟢 RAG Pipeline Active</div>
            <div style="margin-top:4px;">🟢 FAISS Vector DB Ready</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Emergency Hotlines ──────────────────────────────
        st.markdown('<div class="section-header">Emergency Hotlines</div>',
                    unsafe_allow_html=True)
        for name, number in list(EMERGENCY_HOTLINES.items())[:4]:
            st.markdown(f"""
            <div class="hotline-item">
                <span class="hotline-name">{name}</span>
                <span class="hotline-number">{number}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Current Alert Status ─────────────────────────────
        sev = st.session_state.current_severity
        sev_cfg = get_severity_config(sev)
        st.markdown(f"""
        <div style="text-align:center; padding: 8px;
                    background:rgba(255,45,45,0.05); border-radius:4px;
                    border:1px solid rgba(255,45,45,0.15);">
            <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                        color:#484f58; letter-spacing:1px;">ALERT LEVEL</div>
            <div style="font-size:24px; margin:4px 0;">{sev_cfg['emoji']}</div>
            <div class="severity-badge severity-{sev.lower()}">{sev}</div>
        </div>
        """, unsafe_allow_html=True)

        return page


# =============================================================
# HEADER
# =============================================================
def render_header():
    now = datetime.now()
    st.markdown(f"""
    <div class="resqai-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div class="resqai-header-title">RESQAI</div>
                <div class="resqai-header-subtitle">
                    AI Disaster Response &amp; Relief Coordinator
                </div>
            </div>
            <div class="resqai-header-status">
                <div>🟢 SYSTEM OPERATIONAL</div>
                <div style="margin-top:4px; font-size:10px; color:#484f58;">
                    {format_date(now)} &nbsp;|&nbsp;
                    <span id="clock">{format_timestamp(now)}</span>
                </div>
                <div style="margin-top:4px; font-size:10px; color:#484f58;">
                    Docs: {len(st.session_state.uploaded_docs)} &nbsp;|&nbsp;
                    Chunks: {st.session_state.total_chunks}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# PAGE: DASHBOARD
# =============================================================
def page_dashboard():
    st.markdown('<div class="section-header">Emergency Dashboard</div>',
                unsafe_allow_html=True)

    # ── Top Metrics ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    sev = st.session_state.current_severity
    sev_cfg = get_severity_config(sev)

    with col1:
        st.markdown(f"""
        <div class="resq-card">
            <div class="resq-card-title">Alert Level</div>
            <div class="resq-card-value" style="color:{sev_cfg['color']};">
                {sev_cfg['emoji']} {sev}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        dtype = st.session_state.current_disaster
        st.markdown(f"""
        <div class="resq-card">
            <div class="resq-card-title">Disaster Type</div>
            <div class="resq-card-value" style="font-size:18px; padding-top:6px;">
                {dtype}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        shelters = get_nearby_shelters(5)
        total_avail = sum(
            max(0, s["capacity"] - s["current_occupancy"]) for s in shelters
        )
        st.markdown(f"""
        <div class="resq-card">
            <div class="resq-card-title">Available Shelter Spots</div>
            <div class="resq-card-value" style="color:#00bcd4;">{total_avail:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        chat_count = len(st.session_state.chat_history)
        st.markdown(f"""
        <div class="resq-card">
            <div class="resq-card-title">Chat Sessions</div>
            <div class="resq-card-value">{chat_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Quick Emergency Actions</div>',
                    unsafe_allow_html=True)

        emergency_scenarios = [
            ("🏚️ Building Collapse", "There has been a building collapse with people trapped inside. What are immediate response steps?"),
            ("🌊 Flash Flood", "A flash flood is occurring rapidly. What should people do immediately for safety?"),
            ("🔥 Wildfire Nearby", "A wildfire is approaching our area. What evacuation procedures should we follow?"),
            ("⚡ Earthquake Strike", "We just experienced a major earthquake. What are the immediate safety steps?"),
            ("🌀 Hurricane Warning", "A Category 3 hurricane is 12 hours away. What preparations are critical?"),
            ("☢️ Hazmat Incident", "There is a chemical spill near populated areas. What safety protocols apply?"),
        ]

        for i, (label, query) in enumerate(emergency_scenarios):
            if i % 3 == 0:
                cols = st.columns(3)
            if cols[i % 3].button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_query = query
                st.session_state.nav_to_chat = True
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Recent Chat History
        if st.session_state.chat_history:
            st.markdown('<div class="section-header">Recent Activity</div>',
                        unsafe_allow_html=True)
            for msg in st.session_state.chat_history[-4:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:120] + "..."
                icon = "👤" if role == "user" else "🤖"
                color = "#8b949e" if role == "user" else "#00bcd4"
                st.markdown(f"""
                <div style="padding:6px 10px; margin:4px 0;
                            border-left:2px solid {color};
                            font-family:'Rajdhani',sans-serif; font-size:13px;
                            color:#8b949e;">
                    {icon} {content}
                </div>
                """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Nearby Shelters (Preview)</div>',
                    unsafe_allow_html=True)
        for shelter in get_nearby_shelters(3):
            avail = calculate_shelter_availability(shelter)
            pct = avail["occupancy_pct"]
            color = avail["status_color"]
            st.markdown(f"""
            <div class="shelter-card">
                <div class="shelter-name">{shelter['name'][:30]}</div>
                <div class="shelter-type">{shelter['type']}</div>
                <div class="shelter-bar">
                    <div class="shelter-bar-fill"
                         style="width:{pct}%; background:{color};">
                    </div>
                </div>
                <div class="shelter-stat">
                    {avail['available_spots']} spots available &nbsp;·&nbsp;
                    {shelter['distance_km']}km away
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:16px;">All Hotlines</div>',
                    unsafe_allow_html=True)
        for name, number in EMERGENCY_HOTLINES.items():
            st.markdown(f"""
            <div class="hotline-item">
                <span class="hotline-name">{name}</span>
                <span class="hotline-number">{number}</span>
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# PAGE: EMERGENCY CHAT
# =============================================================
def page_chat():
    st.markdown('<div class="section-header">Emergency AI Chat — Powered by RAG</div>',
                unsafe_allow_html=True)

    # ── Status Banner ────────────────────────────────────────
    if st.session_state.vector_store.is_empty():
        st.markdown("""
        <div class="alert-warning">
            ⚠️ No documents uploaded yet. Upload disaster management PDFs in the
            "Document Upload" section to enable RAG-powered answers.
            The chatbot will respond with the fallback message for unknown queries.
        </div>
        """, unsafe_allow_html=True)
    else:
        doc_count = len(st.session_state.uploaded_docs)
        chunk_count = st.session_state.total_chunks
        st.markdown(f"""
        <div class="alert-success">
            ✅ RAG Active — {doc_count} document(s) loaded with {chunk_count} chunks.
            Answers are grounded in your uploaded disaster management documents.
        </div>
        """, unsafe_allow_html=True)

    # ── Chat Display ─────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:40px 20px;
                        color:#484f58; font-family:'Share Tech Mono',monospace; font-size:13px;">
                <div style="font-size:48px; margin-bottom:12px;">🆘</div>
                <div>ResQAI Emergency Chat Ready</div>
                <div style="margin-top:8px; font-size:11px;">
                    Describe your emergency situation or ask about disaster management procedures.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")

                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message chat-user">
                        <div class="chat-avatar chat-avatar-user">👤</div>
                        <div class="chat-content">
                            <div class="chat-meta">YOU &nbsp;·&nbsp; {timestamp}</div>
                            <div>{content}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    severity = msg.get("severity", "UNKNOWN")
                    disaster = msg.get("disaster_type", "")
                    sev_cfg = get_severity_config(severity)
                    meta_extra = ""
                    if disaster and disaster != "Unknown":
                        meta_extra = f"&nbsp;·&nbsp; <span class='severity-badge severity-{severity.lower()}'>{severity}</span> &nbsp; {disaster}"

                    st.markdown(f"""
                    <div class="chat-message chat-assistant">
                        <div class="chat-avatar chat-avatar-assistant">🤖</div>
                        <div class="chat-content">
                            <div class="chat-meta">RESQAI {meta_extra} &nbsp;·&nbsp; {timestamp}</div>
                            <div>{content}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show agent thoughts if available
                    if msg.get("agent_thoughts"):
                        with st.expander("🧠 View Agent Reasoning Chain", expanded=False):
                            for thought in msg["agent_thoughts"]:
                                st.markdown(f"""
                                <div class="thought-step">
                                    <div class="thought-step-title">{thought.step}</div>
                                    <div class="thought-step-content">
                                        {thought.reasoning}<br>
                                        {f'→ {thought.action}' if thought.action else ''}
                                        {f'<br><span style="color:#484f58">{thought.result}</span>' if thought.result else ''}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Show suggested actions
                    if msg.get("suggested_actions"):
                        st.markdown("""
                        <div style="margin:8px 0 4px; font-family:'Share Tech Mono',monospace;
                                    font-size:10px; color:#484f58; letter-spacing:1px;">
                            RECOMMENDED ACTIONS
                        </div>
                        """, unsafe_allow_html=True)
                        for action in msg["suggested_actions"]:
                            st.markdown(f"""
                            <div style="padding:4px 10px; margin:3px 0;
                                        background:rgba(0,188,212,0.06);
                                        border-left:2px solid #00bcd4;
                                        font-family:'Rajdhani',sans-serif; font-size:13px;
                                        color:#90caf9;">
                                ▸ {action}
                            </div>
                            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Input Area ───────────────────────────────────────────
    col_input, col_btn, col_clear = st.columns([6, 1, 1])

    # Handle quick query from dashboard
    default_query = st.session_state.pop("quick_query", "")

    with col_input:
        user_input = st.text_area(
            "Message",
            value=default_query,
            placeholder=(
                "Describe your emergency situation... "
                "(e.g., 'There is a flood approaching our area. What should we do?')"
            ),
            height=80,
            label_visibility="collapsed",
            key="chat_input"
        )

    with col_btn:
        send = st.button("SEND", use_container_width=True, key="send_btn")

    with col_clear:
        if st.button("CLEAR", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Handle Send ──────────────────────────────────────────
    if send and user_input.strip():
        query = user_input.strip()
        timestamp = format_timestamp()

        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "timestamp": timestamp,
        })

        # Quick severity classification
        quick_sev = st.session_state.severity_classifier.classify(query)
        st.session_state.current_severity = quick_sev

        # Retrieve RAG context
        with st.spinner("🔍 Searching documents..."):
            rag_results = st.session_state.vector_store.search(query)
            rag_context = build_rag_context(rag_results)

        # Generate response
        with st.spinner("🤖 ResQAI is analyzing..."):
            try:
                response = st.session_state.agent.chat(
                    user_message=query,
                    rag_context=rag_context,
                    chat_history=st.session_state.chat_history[:-1],
                    weather_data=st.session_state.weather_data,
                    run_analysis=True,
                )

                # Update session state with analysis
                if response.severity:
                    st.session_state.current_severity = response.severity
                if response.disaster_type:
                    st.session_state.current_disaster = response.disaster_type

                # Add assistant message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "timestamp": format_timestamp(),
                    "severity": response.severity or quick_sev,
                    "disaster_type": response.disaster_type,
                    "suggested_actions": response.suggested_actions,
                    "sources_used": response.sources_used,
                    "agent_thoughts": response.agent_thoughts,
                })

            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ System error: {str(e)}. Please check your API key configuration.",
                    "timestamp": format_timestamp(),
                    "severity": "UNKNOWN",
                })

        # Trim history
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY * 2:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY * 2:]

        st.rerun()


# =============================================================
# PAGE: DOCUMENT UPLOAD
# =============================================================
def page_document_upload():
    st.markdown('<div class="section-header">Document Upload & RAG Ingestion</div>',
                unsafe_allow_html=True)

    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        st.markdown("""
        <div class="alert-info">
            📖 Upload disaster management PDFs, emergency response guides, or
            any relevant documents. The AI will use these as its knowledge base
            and answer questions <strong>exclusively</strong> from this content.
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload disaster management PDFs for RAG-powered answers",
        )

        if uploaded_files:
            new_files = []
            for f in uploaded_files:
                file_content = f.read()
                f.seek(0)
                file_hash = compute_file_hash(file_content)
                if file_hash not in st.session_state.vector_store.doc_hashes:
                    new_files.append((f, file_hash))

            if new_files:
                if st.button(
                    f"⬆️ INGEST {len(new_files)} NEW DOCUMENT(S)",
                    use_container_width=True
                ):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, (pdf_file, file_hash) in enumerate(new_files):
                        status_text.markdown(
                            f"<div class='alert-info'>Processing: {pdf_file.name}...</div>",
                            unsafe_allow_html=True
                        )

                        try:
                            # Extract text
                            text = extract_text_from_pdf(pdf_file)
                            if not text.strip():
                                st.warning(f"⚠️ No text found in {pdf_file.name}")
                                continue

                            # Chunk text
                            chunks = chunk_text(text)
                            if not chunks:
                                st.warning(f"⚠️ Could not chunk {pdf_file.name}")
                                continue

                            # Add to vector store
                            with st.spinner(f"🧮 Generating embeddings for {len(chunks)} chunks..."):
                                st.session_state.vector_store.add_documents(
                                    chunks, source_name=pdf_file.name
                                )

                            # Track document
                            st.session_state.vector_store.doc_hashes.add(file_hash)
                            st.session_state.uploaded_docs.append({
                                "name": pdf_file.name,
                                "chunks": len(chunks),
                                "chars": len(text),
                                "timestamp": format_timestamp(),
                            })
                            st.session_state.total_chunks += len(chunks)

                        except Exception as e:
                            st.error(f"❌ Error processing {pdf_file.name}: {e}")

                        progress_bar.progress((i + 1) / len(new_files))

                    status_text.markdown(
                        "<div class='alert-success'>✅ All documents ingested successfully!</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    "<div class='alert-warning'>⚠️ All selected files have already been ingested.</div>",
                    unsafe_allow_html=True
                )

    with col_info:
        st.markdown('<div class="section-header">RAG Pipeline</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="resq-card">
            <div style="font-family:'Share Tech Mono',monospace; font-size:12px; color:#8b949e; line-height:2;">
                <div>📄 1. PDF Upload</div>
                <div style="color:#484f58; margin-left:16px;">↓</div>
                <div>📝 2. Text Extraction (PyPDF)</div>
                <div style="color:#484f58; margin-left:16px;">↓</div>
                <div>✂️ 3. Smart Chunking (1000 chars)</div>
                <div style="color:#484f58; margin-left:16px;">↓</div>
                <div>🧮 4. Gemini Embeddings</div>
                <div style="color:#484f58; margin-left:16px;">↓</div>
                <div>🗄️ 5. FAISS Vector Storage</div>
                <div style="color:#484f58; margin-left:16px;">↓</div>
                <div>🔍 6. Semantic Retrieval</div>
                <div style="color:#484f58; margin-left:16px;">↓</div>
                <div>🤖 7. Gemini LLM Answer</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.uploaded_docs:
            st.markdown('<div class="section-header" style="margin-top:16px;">Ingested Documents</div>',
                        unsafe_allow_html=True)
            for doc in st.session_state.uploaded_docs:
                st.markdown(f"""
                <div class="resq-card">
                    <div style="font-family:'Rajdhani',sans-serif; font-weight:700;
                                font-size:14px; color:#00bcd4;">
                        📄 {doc['name']}
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
                                color:#484f58; margin-top:4px;">
                        {doc['chunks']} chunks &nbsp;·&nbsp;
                        {doc['chars']:,} chars &nbsp;·&nbsp;
                        {doc['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️ CLEAR ALL DOCUMENTS", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.uploaded_docs = []
                st.session_state.total_chunks = 0
                st.rerun()

        # RAG Stats
        st.markdown('<div class="section-header" style="margin-top:16px;">Vector Store Stats</div>',
                    unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Chunks", st.session_state.total_chunks)
        with col_s2:
            st.metric("Documents", len(st.session_state.uploaded_docs))


# =============================================================
# PAGE: FIND SHELTER
# =============================================================
def page_shelters():
    st.markdown('<div class="section-header">Nearby Emergency Shelters & Facilities</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-info">
        📍 Showing nearest emergency facilities. In production, these would be
        real-time geolocated results from emergency management databases.
    </div>
    """, unsafe_allow_html=True)

    filter_type = st.selectbox(
        "Filter by Type",
        ["All", "Emergency Shelter", "Relief Center", "Medical Facility", "Large Shelter"],
    )

    shelters = get_nearby_shelters(10)
    if filter_type != "All":
        shelters = [s for s in shelters if s["type"] == filter_type]

    if not shelters:
        st.info("No facilities found for selected filter.")
        return

    for shelter in shelters:
        avail = calculate_shelter_availability(shelter)

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"""
            <div class="shelter-card">
                <div class="shelter-name">{shelter['name']}</div>
                <div class="shelter-type">{shelter['type']}</div>
                <div style="margin-top:8px; font-family:'Rajdhani',sans-serif;
                            font-size:13px; color:#8b949e;">
                    📍 {shelter['address']}
                </div>
                <div style="margin-top:4px; font-family:'Rajdhani',sans-serif;
                            font-size:13px; color:#8b949e;">
                    📞 {shelter['contact']}
                </div>
                <div style="margin-top:8px;">
                    {''.join(f'<span style="background:rgba(0,188,212,0.1); border:1px solid rgba(0,188,212,0.2); border-radius:2px; padding:2px 6px; margin:2px; font-family:Share Tech Mono,monospace; font-size:10px; color:#00bcd4;">{s}</span>' for s in shelter['services'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            pct = avail["occupancy_pct"]
            color = avail["status_color"]
            st.markdown(f"""
            <div class="shelter-card" style="text-align:center; padding:20px 14px;">
                <div style="font-family:'Bebas Neue',sans-serif; font-size:36px;
                            color:{color};">{pct}%</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                            color:#484f58; letter-spacing:1px;">OCCUPANCY</div>
                <div class="shelter-bar" style="margin:8px 0;">
                    <div class="shelter-bar-fill" style="width:{pct}%; background:{color};"></div>
                </div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:13px; color:#8b949e;">
                    {avail['available_spots']:,} spots free
                </div>
                <div class="severity-badge severity-{'low' if avail['status']=='AVAILABLE' else 'medium' if avail['status']=='MODERATE' else 'high' if avail['status']=='ALMOST FULL' else 'critical'}"
                     style="margin-top:6px;">
                    {avail['status']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="shelter-card" style="text-align:center; padding:20px 14px;">
                <div style="font-family:'Bebas Neue',sans-serif; font-size:32px;
                            color:#00bcd4;">{shelter['distance_km']}</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                            color:#484f58; letter-spacing:1px;">KM AWAY</div>
                <div style="margin-top:8px; font-family:'Rajdhani',sans-serif;
                            font-size:12px; color:#8b949e;">
                    Cap: {shelter['capacity']:,}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# PAGE: WEATHER MONITOR
# =============================================================
def page_weather():
    st.markdown('<div class="section-header">Weather & Environmental Monitor</div>',
                unsafe_allow_html=True)

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        city = st.text_input(
            "City",
            value=st.session_state.weather_city,
            placeholder="Enter city name...",
            label_visibility="collapsed"
        )
    with col_btn:
        fetch_btn = st.button("FETCH", use_container_width=True)

    if fetch_btn or st.session_state.weather_data is None:
        if city.strip():
            with st.spinner("🌤 Fetching weather data..."):
                weather = fetch_weather(city=city.strip())
                st.session_state.weather_data = weather
                st.session_state.weather_city = city.strip()

    weather = st.session_state.weather_data

    if not weather:
        st.markdown("""
        <div class="alert-warning">⚠️ Could not fetch weather data. Check city name or API key.</div>
        """, unsafe_allow_html=True)
        return

    if weather.get("is_mock"):
        st.markdown("""
        <div class="alert-warning">
            🔑 Running in demo mode (no OpenWeather API key). Data shown is simulated.
        </div>
        """, unsafe_allow_html=True)

    # ── Weather Display ──────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        risk = get_weather_risk_level(weather)
        risk_cfg = get_severity_config(risk)
        st.markdown(f"""
        <div class="weather-widget">
            <div style="font-family:'Share Tech Mono',monospace; font-size:11px;
                        color:#484f58; letter-spacing:1px;">
                {weather['city']}, {weather['country']}
            </div>
            <div class="weather-temp">{weather['temp']}°C</div>
            <div class="weather-desc">{weather['description']}</div>
            <div class="weather-detail">Feels like {weather['feels_like']}°C</div>
            <div style="margin-top:12px;">
                <span class="severity-badge severity-{risk.lower()}">{risk_cfg['emoji']} {risk} RISK</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="weather-widget">
        """, unsafe_allow_html=True)
        metrics = [
            ("💧 Humidity", f"{weather['humidity']}%"),
            ("🌬️ Wind", f"{weather['wind_speed_kmh']} km/h"),
            ("☁️ Clouds", f"{weather['cloud_coverage']}%"),
            ("🔍 Visibility", f"{weather['visibility_m']/1000:.1f} km" if weather['visibility_m'] else "N/A"),
            ("📊 Pressure", f"{weather['pressure']} hPa"),
        ]
        for label, value in metrics:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:4px 0;
                        border-bottom:1px solid #21262d;
                        font-family:'Share Tech Mono',monospace; font-size:12px;">
                <span style="color:#484f58;">{label}</span>
                <span style="color:#e6edf3;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">Weather Alerts</div>',
                    unsafe_allow_html=True)
        alerts = weather.get("alerts", [])
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="alert-critical">{alert}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">✅ No active weather alerts for this location.</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:16px;">Temperature Range</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="resq-card" style="text-align:center;">
            <div style="display:flex; justify-content:space-around;">
                <div>
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:28px; color:#2196f3;">
                        {weather['temp_min']}°C
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#484f58;">
                        MIN
                    </div>
                </div>
                <div style="color:#484f58; padding-top:8px;">—</div>
                <div>
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:28px; color:#ff6b00;">
                        {weather['temp_max']}°C
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:10px; color:#484f58;">
                        MAX
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# PAGE: EMERGENCY KIT
# =============================================================
def page_emergency_kit():
    st.markdown('<div class="section-header">Emergency Preparedness Kit Checklist</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-info">
        📦 A well-prepared emergency kit can save lives. Use this checklist to ensure
        you and your family are ready for any disaster situation.
    </div>
    """, unsafe_allow_html=True)

    checklist = build_emergency_kit_checklist()
    cols = st.columns(2)

    category_icons = {
        "Water & Food": "🥤",
        "Medical": "💊",
        "Communication": "📻",
        "Documents": "📄",
        "Clothing & Shelter": "🧥",
        "Tools": "🔧",
    }

    for i, (category, items) in enumerate(checklist.items()):
        with cols[i % 2]:
            icon = category_icons.get(category, "📦")
            st.markdown(f"""
            <div class="resq-card">
                <div style="font-family:'Rajdhani',sans-serif; font-weight:700;
                            font-size:16px; color:#00bcd4; margin-bottom:10px;">
                    {icon} {category}
                </div>
            """, unsafe_allow_html=True)

            for item in items:
                checked = st.checkbox(item, key=f"kit_{category}_{item}")

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="resq-card" style="border-color:rgba(255,45,45,0.3);">
        <div style="font-family:'Bebas Neue',sans-serif; font-size:20px;
                    color:#ff2d2d; letter-spacing:2px; margin-bottom:8px;">
            ⚠️ Critical Reminders
        </div>
        <div style="font-family:'Rajdhani',sans-serif; font-size:14px;
                    color:#8b949e; line-height:1.8;">
            • Store emergency kit in an easily accessible location<br>
            • Review and replenish your kit every 6 months<br>
            • Ensure all family members know where the kit is stored<br>
            • Have a family emergency communication plan<br>
            • Know your local evacuation routes and meeting points<br>
            • Register with local emergency management agencies if you have special needs
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# MAIN APP ROUTER
# =============================================================
def main():
    render_header()
    page = render_sidebar()

    # Handle navigation from quick actions
    if st.session_state.get("nav_to_chat"):
        st.session_state.nav_to_chat = False
        page = "💬 Emergency Chat"

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "💬 Emergency Chat":
        page_chat()
    elif page == "📚 Document Upload":
        page_document_upload()
    elif page == "🏥 Find Shelter":
        page_shelters()
    elif page == "🌤 Weather Monitor":
        page_weather()
    elif page == "📋 Emergency Kit":
        page_emergency_kit()


if __name__ == "__main__":
    main()
