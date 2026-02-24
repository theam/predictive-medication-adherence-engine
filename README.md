# 🥇 Predictive Medication Adherence Engine

AI-powered system for predicting medication non-adherence risk and orchestrating personalized interventions to improve patient outcomes.

## 📋 Table of Contents

- [Overview](#overview)
- [Business Value](#business-value)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Testing](#testing)

## Overview

The Predictive Medication Adherence Engine (PMAE) is a comprehensive solution for OptumRx to combat medication non-adherence, which costs **$528 billion annually** in the US healthcare system.

### Key Capabilities

- **Risk Prediction**: ML model (XGBoost) analyzing 50+ features to predict non-adherence risk
- **Multi-Channel Interventions**: Automated, personalized outreach via SMS, email, voice, chatbot, and care manager escalation
- **Conversational AI**: Claude-powered chatbot for barrier identification and resolution
- **Analytics & ROI**: Closed-loop tracking of intervention effectiveness and cost savings

## Business Value

| Metric | Value |
|--------|-------|
| **Annual US cost of non-adherence** | $528 billion |
| **Savings per adherent diabetic patient** | $3,000 - $8,000/year |
| **Pharmacy revenue increase** | 5-10% more refills |
| **Projected Year 1 ROI** | 300-500% |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  Claims Data │ Pharmacy Transactions │ Member Demographics       │
│                    ↓ Feature Store (Feast/Tecton)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ML LAYER                                  │
│  Risk Prediction (XGBoost) │ Channel Selector │ Timing Optimizer │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                            │
│  SMS Service │ Voice Service │ Chatbot │ Email │ Care Manager    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   CONVERSATIONAL LAYER                           │
│  LLM Agent (Claude) │ Medication Knowledge RAG │ Escalation      │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Risk Prediction Engine

```python
from src.models import AdherenceRiskPredictor

predictor = AdherenceRiskPredictor()
prediction = predictor.predict_single(
    patient_id="P001",
    patient_data={"age": 55, "diagnosis_codes": ["E11.9"]},
    fills=medication_history
)

print(f"Risk Score: {prediction.risk_score}/100")
print(f"Risk Level: {prediction.risk_level}")
print(f"Top Factors: {prediction.top_factors}")
```

**Analyzed Features (50+):**
- Refill history (gaps, PDC, consistency)
- Demographics (age, plan type)
- Medication complexity (polypharmacy, regimen)
- Cost factors (copay, trends)
- Clinical risk (depression, chronic conditions)
- Behavioral (channel preferences, engagement)

### 2. Multi-Channel Intervention Orchestrator

```python
from src.services import InterventionOrchestrator

orchestrator = InterventionOrchestrator()

# Automatically triggered by risk prediction
response = await orchestrator.process_risk_prediction(
    prediction=risk_prediction,
    patient_data=patient_profile
)

# Or explicitly create intervention
from src.models.schemas import InterventionRequest

request = InterventionRequest(
    patient_id="P001",
    channel="sms",
    priority=1
)
response = await orchestrator.create_intervention(request, patient_data)
```

**Supported Channels:**
- 📱 SMS / Push Notifications
- 📧 Email Campaigns
- 📞 Voice (IVR)
- 💬 Chatbot
- 👤 Care Manager Escalation

### 3. Conversational AI Agent

```python
from src.services import ConversationalAgent
from src.models.schemas import ChatRequest

agent = ConversationalAgent(anthropic_api_key="...")

request = ChatRequest(
    patient_id="P001",
    message="The medication is too expensive for me"
)

response = await agent.chat(request, patient_context)
# Identifies COST barrier, suggests copay assistance programs
```

**Capabilities:**
- Barrier detection (cost, forgetfulness, side effects, complexity)
- Medication education
- Copay assistance program lookup
- Smart escalation to pharmacists

### 4. Population Management Dashboard

Interactive Streamlit dashboard with:
- Population risk distribution
- High-risk patient list with bulk actions
- Intervention tracking and effectiveness
- ROI calculator

### 5. Closed-Loop Analytics

```python
from src.services import AnalyticsEngine

analytics = AnalyticsEngine()

# Track intervention outcomes
analytics.record_intervention_outcome(
    intervention_id="I001",
    patient_id="P001",
    channel="sms",
    status="successful",
    refill_completed=True
)

# Get channel effectiveness
effectiveness = analytics.get_channel_effectiveness(lookback_days=90)

# Calculate ROI
roi = analytics.calculate_roi(start_date, end_date)
print(f"ROI: {roi.roi_percentage}%")
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip or uv package manager

### Installation

```bash
# Clone repository
cd predictive-medication-adherence-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or with optional dependencies
pip install -e ".[dev,dashboard]"
```

### Generate Demo Data

```bash
python data/generate_synthetic_data.py
```

### Start API Server

```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Or use the CLI
python -m src.api.main
```

### Start Dashboard

```bash
streamlit run src/dashboard/app.py
```

## API Documentation

Once the server is running, access the interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predictions/risk` | POST | Get risk prediction for a patient |
| `/api/v1/predictions/batch` | POST | Batch predictions for multiple patients |
| `/api/v1/interventions` | POST | Create and send an intervention |
| `/api/v1/chat` | POST | Interact with conversational AI |
| `/api/v1/analytics/roi` | GET | Calculate ROI metrics |
| `/api/v1/analytics/channels` | GET | Get channel effectiveness |

### Example: Risk Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predictions/risk \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "prediction_horizon_days": 30
  }'
```

Response:
```json
{
  "patient_id": "P001",
  "risk_score": 72.5,
  "risk_level": "high",
  "top_risk_factors": [
    {
      "factor_name": "pdc_90_days",
      "impact_score": 0.35,
      "description": "Medication coverage in last 90 days"
    }
  ],
  "confidence_score": 0.85
}
```

## Configuration

### Environment Variables

Create a `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Claude API (for conversational agent)
ANTHROPIC_API_KEY=sk-ant-...

# Database (production)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/adherence

# Redis (for caching/queues)
REDIS_URL=redis://localhost:6379/0

# Twilio (for SMS/Voice)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
```

### Feature Flags

```env
FF_ENABLE_SMS_INTERVENTIONS=true
FF_ENABLE_VOICE_INTERVENTIONS=false
FF_ENABLE_CHATBOT=true
FF_ENABLE_AB_TESTING=true
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_risk_predictor.py -v
```

## Project Structure

```
predictive-medication-adherence-engine/
├── src/
│   ├── models/
│   │   ├── schemas.py          # Pydantic models
│   │   ├── feature_engineer.py # Feature engineering
│   │   └── risk_predictor.py   # XGBoost risk model
│   ├── services/
│   │   ├── intervention_orchestrator.py
│   │   ├── channel_selector.py
│   │   ├── message_templates.py
│   │   ├── conversational_agent.py
│   │   └── analytics_engine.py
│   ├── api/
│   │   └── main.py             # FastAPI application
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
├── data/
│   └── generate_synthetic_data.py
├── tests/
├── config/
│   └── settings.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost / scikit-learn |
| API Framework | FastAPI |
| LLM | Claude (Anthropic) |
| Dashboard | Streamlit + Plotly |
| Database | PostgreSQL + SQLAlchemy |
| Caching | Redis |
| Task Queue | Celery |
| SMS/Voice | Twilio |

## Roadmap

- [x] Risk prediction model
- [x] Multi-channel orchestrator
- [x] Conversational AI agent
- [x] Population dashboard
- [x] Analytics engine
- [ ] Vector DB for medication RAG
- [ ] Reinforcement learning for channel optimization
- [ ] Real-time event streaming
- [ ] FHIR integration
- [ ] Production Kubernetes deployment

## License

MIT License - See LICENSE file for details.

---

Built with ❤️ for OptumRx Innovation Team
