# =============================================================
# ResQAI — Agentic AI Module
# Multi-step reasoning agent for disaster response coordination
# =============================================================

import os
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import google.generativeai as genai

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE,
    SEVERITY_LEVELS, DISASTER_TYPES
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

FALLBACK_MESSAGE = (
    "I could not find this information in the uploaded disaster "
    "management documents. Please upload relevant PDFs or consult "
    "official emergency management resources."
)


# ─── Data Classes ─────────────────────────────────────────

@dataclass
class AgentThought:
    """Represents one reasoning step in the agent's chain of thought."""
    step: str
    reasoning: str
    action: Optional[str] = None
    result: Optional[str] = None


@dataclass
class DisasterAnalysis:
    """Structured output from disaster analysis."""
    disaster_type: str = "Unknown"
    severity: str = "UNKNOWN"
    immediate_threats: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    priority_resources: List[str] = field(default_factory=list)
    reasoning_chain: List[AgentThought] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ChatResponse:
    """Structured chat response with metadata."""
    answer: str
    sources_used: bool = False
    disaster_type: Optional[str] = None
    severity: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    agent_thoughts: List[AgentThought] = field(default_factory=list)


# ─── Gemini LLM Wrapper ───────────────────────────────────

class GeminiLLM:
    """Wrapper around Google Gemini for structured generation."""

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=2048,
            )
        )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            Generated text response
        """
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict:
        """
        Generate a JSON-structured response.

        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system instruction

        Returns:
            Parsed JSON dict or empty dict on failure
        """
        json_system = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no backticks, no explanation. "
            "Just raw JSON."
        )
        raw = self.generate(prompt, json_system)
        # Strip any accidental markdown
        raw = re.sub(r'```json\s*|\s*```', '', raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            return {}


# ─── Disaster Analysis Agent ──────────────────────────────

class DisasterAnalysisAgent:
    """
    Agentic AI that performs multi-step reasoning to analyze
    disaster situations and generate response strategies.
    """

    def __init__(self):
        self.llm = GeminiLLM()

    def analyze(self, user_message: str,
                weather_data: Optional[Dict] = None) -> DisasterAnalysis:
        """
        Perform multi-step analysis of a disaster situation.

        Args:
            user_message: User's description of the situation
            weather_data: Optional current weather data

        Returns:
            DisasterAnalysis with full reasoning chain
        """
        analysis = DisasterAnalysis()
        thoughts = []

        # ── Step 1: Identify Disaster Type ────────────────
        thought1 = AgentThought(
            step="1. Disaster Type Identification",
            reasoning="Analyzing the situation to identify disaster category"
        )
        disaster_json = self.llm.generate_json(
            prompt=f"""
Analyze this emergency situation and identify the disaster type.

Situation: {user_message}
Weather context: {json.dumps(weather_data) if weather_data else "Not available"}

Available disaster types: {', '.join(DISASTER_TYPES)}

Return JSON with:
{{
  "disaster_type": "one of the available types",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
""",
            system_prompt="You are an expert emergency management classifier."
        )
        analysis.disaster_type = disaster_json.get("disaster_type", "Unknown")
        analysis.confidence = disaster_json.get("confidence", 0.5)
        thought1.action = f"Classified as: {analysis.disaster_type}"
        thought1.result = disaster_json.get("reasoning", "")
        thoughts.append(thought1)

        # ── Step 2: Severity Assessment ───────────────────
        thought2 = AgentThought(
            step="2. Severity Assessment",
            reasoning="Evaluating threat level and immediate danger"
        )
        severity_json = self.llm.generate_json(
            prompt=f"""
Assess the severity of this {analysis.disaster_type} disaster.

Situation: {user_message}
Weather: {json.dumps(weather_data) if weather_data else "Unknown"}

Return JSON with:
{{
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "immediate_threats": ["threat1", "threat2", "threat3"],
  "reasoning": "explanation of severity assessment"
}}
""",
            system_prompt="You are an expert emergency management risk assessor."
        )
        analysis.severity = severity_json.get("severity", "UNKNOWN")
        analysis.immediate_threats = severity_json.get("immediate_threats", [])
        thought2.action = f"Severity: {analysis.severity}"
        thought2.result = severity_json.get("reasoning", "")
        thoughts.append(thought2)

        # ── Step 3: Response Strategy Generation ──────────
        thought3 = AgentThought(
            step="3. Response Strategy Generation",
            reasoning="Generating prioritized emergency response actions"
        )
        strategy_json = self.llm.generate_json(
            prompt=f"""
Generate an emergency response strategy for this situation.

Disaster Type: {analysis.disaster_type}
Severity: {analysis.severity}
Immediate Threats: {', '.join(analysis.immediate_threats)}
Situation: {user_message}

Return JSON with:
{{
  "recommended_actions": ["action1", "action2", "action3", "action4", "action5"],
  "priority_resources": ["resource1", "resource2", "resource3"],
  "reasoning": "strategy explanation"
}}
""",
            system_prompt=(
                "You are an expert emergency response coordinator with "
                "20+ years experience in disaster management."
            )
        )
        analysis.recommended_actions = strategy_json.get("recommended_actions", [])
        analysis.priority_resources = strategy_json.get("priority_resources", [])
        thought3.action = "Generated response strategy"
        thought3.result = strategy_json.get("reasoning", "")
        thoughts.append(thought3)

        analysis.reasoning_chain = thoughts
        return analysis


# ─── RAG Chat Agent ───────────────────────────────────────

class RAGChatAgent:
    """
    Chat agent that answers questions using RAG context.
    Strictly uses retrieved document context for answers.
    """

    SYSTEM_PROMPT = """You are ResQAI, an expert AI assistant for disaster response and emergency management.

Your role is to:
1. Provide accurate, life-saving information based ONLY on the provided document context
2. Classify disaster types and severity
3. Give clear, actionable emergency guidance
4. Prioritize human safety above all else

STRICT RULES:
- Answer ONLY from the provided context
- If context is insufficient, say: "I could not find this information in the uploaded disaster management documents."
- Be concise and clear in emergencies
- Always recommend professional emergency services for life-threatening situations
- Never make up statistics, procedures, or contact information"""

    def __init__(self):
        self.llm = GeminiLLM()
        self.analysis_agent = DisasterAnalysisAgent()

    def chat(self,
             user_message: str,
             rag_context: str,
             chat_history: List[Dict],
             weather_data: Optional[Dict] = None,
             run_analysis: bool = True) -> ChatResponse:
        """
        Generate a chat response using RAG context.

        Args:
            user_message: The user's question or emergency description
            rag_context: Retrieved document chunks as context
            chat_history: Previous conversation turns
            weather_data: Optional current weather info
            run_analysis: Whether to run full disaster analysis

        Returns:
            ChatResponse with answer and metadata
        """
        response = ChatResponse(answer="")

        # ── Run Disaster Analysis ──────────────────────────
        analysis = None
        if run_analysis:
            try:
                analysis = self.analysis_agent.analyze(user_message, weather_data)
                response.disaster_type = analysis.disaster_type
                response.severity = analysis.severity
                response.suggested_actions = analysis.recommended_actions[:3]
                response.agent_thoughts = analysis.reasoning_chain
            except Exception as e:
                print(f"Analysis agent error: {e}")

        # ── Build Chat History String ──────────────────────
        history_str = ""
        for turn in chat_history[-6:]:  # Last 3 exchanges
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_str += f"\n{role.upper()}: {content}"

        # ── Build RAG Prompt ───────────────────────────────
        has_context = bool(rag_context and rag_context.strip())
        response.sources_used = has_context

        if has_context:
            prompt = f"""
DOCUMENT CONTEXT (from uploaded disaster management PDFs):
{rag_context}

CONVERSATION HISTORY:
{history_str if history_str else "No previous conversation."}

CURRENT SITUATION ANALYSIS:
- Disaster Type: {analysis.disaster_type if analysis else "Unknown"}
- Severity: {analysis.severity if analysis else "Unknown"}
- Weather: {_format_weather(weather_data) if weather_data else "Not available"}

USER QUESTION: {user_message}

Instructions:
1. Answer using ONLY the provided document context above
2. If the context doesn't contain the answer, respond with: "I could not find this information in the uploaded disaster management documents."
3. Be specific and actionable
4. Cite which part of the context supports your answer
5. For life-threatening situations, always recommend calling 911 first
"""
        else:
            prompt = f"""
NO DOCUMENT CONTEXT AVAILABLE.

CONVERSATION HISTORY:
{history_str if history_str else "No previous conversation."}

USER QUESTION: {user_message}

Since no disaster management documents have been uploaded, respond with:
"I could not find this information in the uploaded disaster management documents."

Then briefly suggest what type of documents would help answer this question.
"""

        response.answer = self.llm.generate(prompt, self.SYSTEM_PROMPT)
        return response


# ─── Severity Classifier ──────────────────────────────────

class SeverityClassifier:
    """Lightweight classifier for quick severity assessment."""

    KEYWORDS = {
        "CRITICAL": [
            "trapped", "unconscious", "not breathing", "cardiac arrest",
            "building collapse", "mass casualty", "tsunami warning",
            "nuclear", "chemical leak", "explosion", "fire spreading",
            "multiple deaths", "no evacuation possible"
        ],
        "HIGH": [
            "injured", "evacuate", "flooding", "fire", "earthquake",
            "hurricane", "tornado warning", "gas leak", "structural damage",
            "power outage", "road blocked", "missing person"
        ],
        "MEDIUM": [
            "warning", "watch", "prepare", "shelter", "supplies",
            "advisory", "potential", "monitoring", "precaution"
        ],
        "LOW": [
            "update", "information", "guidance", "planning", "training",
            "drill", "exercise", "awareness", "education"
        ]
    }

    def classify(self, text: str) -> str:
        """
        Quickly classify severity based on keyword matching.

        Args:
            text: Text to classify

        Returns:
            Severity level string
        """
        text_lower = text.lower()
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if any(kw in text_lower for kw in self.KEYWORDS[level]):
                return level
        return "UNKNOWN"


# ─── Helper Functions ─────────────────────────────────────

def _format_weather(weather_data: Dict) -> str:
    """Format weather data for prompt injection."""
    if not weather_data:
        return "Unknown"
    temp = weather_data.get("temp", "N/A")
    desc = weather_data.get("description", "N/A")
    wind = weather_data.get("wind_speed", "N/A")
    humidity = weather_data.get("humidity", "N/A")
    return (
        f"Temp: {temp}°C, Conditions: {desc}, "
        f"Wind: {wind}m/s, Humidity: {humidity}%"
    )
