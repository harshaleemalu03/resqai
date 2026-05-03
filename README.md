<<<<<<< HEAD
# 🆘 ResQAI — AI Disaster Response & Relief Coordinator

<div align="center">

![ResQAI](https://img.shields.io/badge/ResQAI-Disaster%20Response%20AI-red?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMNCAyMGgxNkwxMiAyem0wIDNsNi41IDE0aC0xM0wxMiA1em0tMSA2djRoMnYtNGgtMnptMCA2djJoMnYtMmgtMnoiIGZpbGw9IndoaXRlIi8+PC9zdmc+)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?style=for-the-badge&logo=google)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange?style=for-the-badge)

**Agentic AI + RAG platform for disaster response and emergency assistance**

</div>

---

## 🎯 Overview

ResQAI is a production-ready AI system that combines **Retrieval-Augmented Generation (RAG)**, **Agentic AI reasoning**, and **real-time weather data** to provide intelligent disaster response coordination. The system answers emergency questions exclusively from uploaded disaster management documents, ensuring accurate and grounded responses.

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Emergency Chatbot** | Gemini-powered chatbot with multi-step agent reasoning |
| 📚 **RAG Pipeline** | Upload PDFs → Extract → Chunk → Embed → FAISS → Retrieve → Answer |
| 🧠 **Agentic AI** | Identifies disaster type, assesses severity, generates response strategies |
| 🌤 **Weather Integration** | Real-time weather monitoring with disaster risk assessment |
| 🏥 **Shelter Finder** | Nearby emergency shelters with real-time availability |
| 📊 **Emergency Dashboard** | Live severity indicators, alert cards, quick actions |
| 💬 **Chat History** | Persistent conversation with agent reasoning chain visibility |
| 🎨 **Dark Modern UI** | Military-grade emergency response aesthetic |

## 🏗️ Architecture

```
ResQAI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment configuration template
├── create_sample_pdfs.py           # Sample PDF generator
├── sample_docs/                    # Sample disaster management docs
│   └── disaster_management_guide.txt
└── src/
    ├── config.py                   # Central configuration
    ├── agents/
    │   └── responder.py            # Agentic AI (DisasterAnalysisAgent, RAGChatAgent)
    ├── rag/
    │   └── pipeline.py             # RAG pipeline (FAISS, embeddings, chunking)
    └── utils/
        ├── weather.py              # OpenWeatherMap integration
        └── helpers.py              # Shared utilities
```

## 🔄 RAG Pipeline Flow

```
PDF Upload → Text Extraction (PyPDF)
           → Smart Chunking (1000 chars, 200 overlap)
           → Gemini Embeddings (models/embedding-001)
           → FAISS Vector Storage (cosine similarity)
           → Semantic Retrieval (top-5 chunks)
           → Context-grounded LLM Answer
```

## 🤖 Agentic Reasoning Chain

For each user query, the agent executes:

1. **Step 1 — Disaster Identification**: Classify disaster type from 16 categories
2. **Step 2 — Severity Assessment**: Evaluate CRITICAL/HIGH/MEDIUM/LOW risk level
3. **Step 3 — Response Strategy**: Generate prioritized emergency actions

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/resqai.git
cd resqai
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

**Getting API Keys:**
- **Gemini**: [Google AI Studio](https://aistudio.google.com/app/apikey) (free)
- **OpenWeather**: [OpenWeatherMap](https://openweathermap.org/api) (free tier available)

### 5. Generate Sample PDFs (Optional)

```bash
pip install reportlab
python create_sample_pdfs.py
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📋 Usage Guide

### Uploading Documents
1. Go to **📚 Document Upload** in the sidebar
2. Click the file uploader and select PDF(s)
3. Click **INGEST DOCUMENTS**
4. Wait for embedding generation to complete

### Using the Emergency Chat
1. Navigate to **💬 Emergency Chat**
2. Describe your emergency situation
3. The AI will:
   - Search your uploaded documents (RAG)
   - Identify disaster type and severity
   - Generate a grounded, actionable response
   - Show agent reasoning chain

### Quick Emergency Actions
- Use the **Dashboard** quick-action buttons for common scenarios
- Click any scenario to instantly query the AI

## 🌡️ Severity Levels

| Level | Color | Description |
|---|---|---|
| 🔴 CRITICAL | Red | Immediate life threat, mass casualties |
| 🟠 HIGH | Orange | Serious injury risk, evacuation needed |
| 🟡 MEDIUM | Yellow | Elevated risk, preparation required |
| 🟢 LOW | Green | Informational, awareness level |

## 🔧 Configuration

Key settings in `src/config.py`:

```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
MAX_RETRIEVAL_DOCS = 5   # Documents retrieved per query
TEMPERATURE = 0.3        # LLM temperature (lower = more focused)
GEMINI_MODEL = "gemini-1.5-flash"
```

## 🐳 Deployment — Hugging Face Spaces

### 1. Create a Hugging Face Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- Choose **Streamlit** as the SDK
- Set visibility (Public or Private)

### 2. Create `packages.txt`
```
libgl1-mesa-glx
```

### 3. Add Secrets
In Space Settings → Repository secrets:
- `GEMINI_API_KEY` = your Gemini key
- `OPENWEATHER_API_KEY` = your OpenWeather key

### 4. Push to Hugging Face

```bash
git remote add space https://huggingface.co/spaces/yourusername/resqai
git push space main
```

### 5. Access Your Space
Your app will be live at:
`https://huggingface.co/spaces/yourusername/resqai`

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit 1.32 |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Gemini Embedding-001 |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| RAG Framework | LangChain + Custom Pipeline |
| Weather API | OpenWeatherMap |
| PDF Processing | PyPDF |
| Language | Python 3.10+ |

## 📁 Key Files Reference

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, page routing, session management |
| `src/config.py` | All constants, API keys, mock data |
| `src/agents/responder.py` | Gemini LLM, DisasterAnalysisAgent, RAGChatAgent |
| `src/rag/pipeline.py` | FAISSVectorStore, chunking, embedding, retrieval |
| `src/utils/weather.py` | OpenWeatherMap integration, alert generation |
| `src/utils/helpers.py` | Shelter data, checklist, formatting utilities |

## ⚠️ Disclaimer

ResQAI is an AI-assisted tool for informational purposes. In real emergencies:
1. **Always call 911 first**
2. Follow official emergency management guidance
3. This tool supplements but does not replace professional emergency response

## 📄 License

MIT License — Free to use, modify, and distribute.

---

<div align="center">
Built for disaster preparedness and emergency response coordination
</div>
=======
# resqai
>>>>>>> 7406ac257b9df2b2486e322d3a55224b5fe1072b
