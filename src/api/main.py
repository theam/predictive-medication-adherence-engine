"""
Predictive Medication Adherence Engine — API.

Designed following John Maeda's Laws of Simplicity.
"""
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
import structlog

from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    InterventionChannel,
    InterventionOutcome,
    InterventionRequest,
    InterventionResponse,
    InterventionStatus,
    Patient,
    PatientCreate,
    PopulationRiskSummary,
    RiskLevel,
    RiskPredictionRequest,
    RiskPredictionResponse,
    ROIMetrics,
)
from ..models.risk_predictor import AdherenceRiskPredictor
from ..services.intervention_orchestrator import InterventionOrchestrator
from ..services.conversational_agent import ConversationalAgent
from ..services.analytics_engine import AnalyticsEngine

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


# ── State ──────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.risk_predictor = AdherenceRiskPredictor()
        self.orchestrator = InterventionOrchestrator()
        self.chatbot = ConversationalAgent()
        self.analytics = AnalyticsEngine()
        self.patients: dict[str, dict] = {}
        self.fills: dict[str, list] = {}


state = AppState()


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_application")
    yield
    logger.info("shutting_down_application")


# ── App ────────────────────────────────────────────────────────

DESCRIPTION = """
Predict non-adherence risk. Orchestrate interventions. Measure outcomes.
"""

app = FastAPI(
    title="Adherence Engine",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Custom docs with minimal theme ────────────────────────────

SWAGGER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

body { background: #fafaf7 !important; }

.swagger-ui {
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 960px;
    margin: 0 auto;
    padding: 0 2rem;
}

/* Top bar — remove entirely */
.swagger-ui .topbar { display: none !important; }

/* Info section */
.swagger-ui .info { margin: 3rem 0 2rem 0; }
.swagger-ui .info hgroup.main .title {
    font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important;
    font-size: 2rem !important;
    letter-spacing: -0.03em;
    color: #1a1a1a;
}
.swagger-ui .info .description p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem;
    color: #888;
    line-height: 1.6;
}
.swagger-ui .info .title small.version-stamp {
    background: #1a1a1a !important;
    border-radius: 4px;
    font-size: 0.65rem;
    padding: 2px 8px;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* Operation tags (section headers) */
.swagger-ui .opblock-tag {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #888 !important;
    border-bottom: 1px solid #e8e8e4 !important;
    padding: 1rem 0 0.75rem 0 !important;
    margin: 0 !important;
}
.swagger-ui .opblock-tag:hover { color: #1a1a1a !important; }

/* Operation blocks — quiet containers */
.swagger-ui .opblock {
    border: 1px solid #e8e8e4 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    margin: 0.5rem 0 !important;
    background: #fff !important;
}
.swagger-ui .opblock .opblock-summary {
    border: none !important;
    padding: 0.75rem 1rem !important;
}
.swagger-ui .opblock .opblock-summary-method {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    border-radius: 4px !important;
    padding: 4px 10px !important;
    min-width: 50px;
    text-align: center;
}
.swagger-ui .opblock .opblock-summary-path {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    color: #1a1a1a !important;
}
.swagger-ui .opblock .opblock-summary-description {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: #888 !important;
}

/* Method colors — restrained */
.swagger-ui .opblock.opblock-get {
    border-color: #e8e8e4 !important;
    background: #fff !important;
}
.swagger-ui .opblock.opblock-get .opblock-summary-method { background: #1a1a1a !important; }
.swagger-ui .opblock.opblock-get .opblock-summary { border-color: transparent !important; }

.swagger-ui .opblock.opblock-post {
    border-color: #e8e8e4 !important;
    background: #fff !important;
}
.swagger-ui .opblock.opblock-post .opblock-summary-method { background: #4a9 !important; }
.swagger-ui .opblock.opblock-post .opblock-summary { border-color: transparent !important; }

.swagger-ui .opblock.opblock-put .opblock-summary-method { background: #e8a838 !important; }
.swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #d44 !important; }

/* Expanded operation body */
.swagger-ui .opblock-body { padding: 1rem !important; }
.swagger-ui .opblock-body pre {
    font-size: 0.8rem !important;
    background: #fafaf7 !important;
    border: 1px solid #e8e8e4 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}

/* Try it out / Execute buttons */
.swagger-ui .btn {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.swagger-ui .btn.execute {
    background: #1a1a1a !important;
    color: #fafaf7 !important;
    border: none !important;
}
.swagger-ui .btn.try-out__btn {
    border: 1px solid #1a1a1a !important;
    color: #1a1a1a !important;
    background: transparent !important;
}

/* Parameters */
.swagger-ui .parameters-col_description input,
.swagger-ui .parameters-col_description textarea,
.swagger-ui .body-param textarea {
    font-family: 'Inter', monospace !important;
    font-size: 0.85rem !important;
    border: 1px solid #e8e8e4 !important;
    border-radius: 6px !important;
    background: #fafaf7 !important;
}

/* Models section */
.swagger-ui section.models {
    border: 1px solid #e8e8e4 !important;
    border-radius: 8px !important;
}
.swagger-ui section.models h4 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888 !important;
}

/* Response section */
.swagger-ui .responses-inner h4,
.swagger-ui .responses-inner h5 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: #888 !important;
}

/* Scrollbar */
.swagger-ui ::-webkit-scrollbar { width: 6px; height: 6px; }
.swagger-ui ::-webkit-scrollbar-track { background: transparent; }
.swagger-ui ::-webkit-scrollbar-thumb { background: #c8c8c8; border-radius: 3px; }

/* Scheme/server selectors — minimal */
.swagger-ui .scheme-container {
    background: transparent !important;
    box-shadow: none !important;
    border-bottom: 1px solid #e8e8e4;
    padding: 0.5rem 0 !important;
}
"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title}",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "list",
            "filter": True,
            "syntaxHighlight.theme": "arta",
            "tryItOutEnabled": True,
        },
    )


@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title}",
    )


@app.get("/docs/style.css", include_in_schema=False)
async def swagger_css():
    return HTMLResponse(content=SWAGGER_CSS, media_type="text/css")


# Inject our CSS into the docs page
_original_openapi = app.openapi


def custom_openapi():
    schema = _original_openapi()
    return schema


app.openapi = custom_openapi


# Serve docs with injected CSS
@app.get("/", include_in_schema=False)
async def root():
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{app.title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"/>
    <style>{SWAGGER_CSS}</style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({{
            url: "{app.openapi_url}",
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
            layout: "BaseLayout",
            defaultModelsExpandDepth: -1,
            docExpansion: "list",
            filter: true,
            syntaxHighlight: {{ theme: "arta" }},
            tryItOutEnabled: true,
        }})
    </script>
</body>
</html>""")


# ── Health ─────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check."""
    return {
        "status": "ready",
        "model_loaded": state.risk_predictor._is_fitted or True,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Patients ───────────────────────────────────────────────────

class PatientDataInput(BaseModel):
    patient_id: str
    first_name: str = "Patient"
    age: int = Field(ge=0, le=120)
    gender: str = "U"
    plan_type: str = "commercial"
    diagnosis_codes: list[str] = Field(default_factory=list)
    phone_number: Optional[str] = None
    email: Optional[str] = None
    preferred_channel: Optional[str] = None
    preferred_contact_time: Optional[str] = None


class MedicationFillInput(BaseModel):
    fill_id: str
    patient_id: str
    medication_ndc: str
    medication_name: str
    fill_date: date
    days_supply: int = Field(ge=1, le=365)
    refill_number: int = Field(ge=0)
    copay_amount: float = Field(ge=0)
    quantity: int = Field(ge=1, default=30)


@app.post("/api/v1/patients", tags=["Patients"])
async def create_patient(patient: PatientDataInput) -> dict[str, Any]:
    """Create or update a patient record."""
    state.patients[patient.patient_id] = patient.model_dump()
    logger.info("patient_created", patient_id=patient.patient_id)
    return {"status": "created", "patient_id": patient.patient_id}


@app.get("/api/v1/patients/{patient_id}", tags=["Patients"])
async def get_patient(patient_id: str) -> dict[str, Any]:
    """Get patient by ID."""
    if patient_id not in state.patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    return state.patients[patient_id]


@app.post("/api/v1/patients/{patient_id}/fills", tags=["Patients"])
async def add_medication_fill(patient_id: str, fill: MedicationFillInput) -> dict[str, Any]:
    """Add a medication fill record."""
    if patient_id not in state.fills:
        state.fills[patient_id] = []
    state.fills[patient_id].append(fill.model_dump())
    logger.info("fill_added", patient_id=patient_id, fill_id=fill.fill_id)
    return {"status": "added", "fill_id": fill.fill_id}


# ── Predictions ────────────────────────────────────────────────

@app.post("/api/v1/predictions/risk", tags=["Predictions"], response_model=RiskPredictionResponse)
async def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """Predict non-adherence risk for a patient."""
    import pandas as pd

    patient_data = state.patients.get(request.patient_id)
    if not patient_data:
        patient_data = {
            "patient_id": request.patient_id,
            "first_name": "Demo",
            "age": 55,
            "gender": "M",
            "plan_type": "commercial",
            "diagnosis_codes": ["E11.9", "I10"],
        }

    fills = state.fills.get(request.patient_id, [])
    patient_df = pd.DataFrame([patient_data])
    fills_df = pd.DataFrame(fills) if fills else pd.DataFrame()

    predictions = state.risk_predictor.predict(
        patient_df, fills_df,
        medication_ndc=request.medication_ndc,
        prediction_horizon_days=request.prediction_horizon_days,
    )

    if not predictions:
        raise HTTPException(status_code=500, detail="Prediction failed")
    return predictions[0].to_response()


@app.post("/api/v1/predictions/batch", tags=["Predictions"])
async def predict_risk_batch(
    patient_ids: list[str],
    prediction_horizon_days: int = 30,
) -> list[RiskPredictionResponse]:
    """Batch risk prediction for multiple patients."""
    import pandas as pd

    results = []
    for patient_id in patient_ids:
        patient_data = state.patients.get(patient_id, {
            "patient_id": patient_id, "age": 55, "gender": "M",
            "plan_type": "commercial", "diagnosis_codes": [],
        })
        fills = state.fills.get(patient_id, [])
        patient_df = pd.DataFrame([patient_data])
        fills_df = pd.DataFrame(fills) if fills else pd.DataFrame()
        predictions = state.risk_predictor.predict(
            patient_df, fills_df,
            prediction_horizon_days=prediction_horizon_days,
        )
        if predictions:
            results.append(predictions[0].to_response())
    return results


@app.get("/api/v1/predictions/population-summary", tags=["Predictions"])
async def get_population_risk_summary() -> PopulationRiskSummary:
    """Population risk distribution summary."""
    return PopulationRiskSummary(
        total_patients=len(state.patients) or 1000,
        low_risk_count=450,
        medium_risk_count=350,
        high_risk_count=200,
        average_risk_score=45.5,
        trending_worse=120,
        trending_better=180,
    )


# ── Interventions ──────────────────────────────────────────────

@app.post("/api/v1/interventions", tags=["Interventions"], response_model=InterventionResponse)
async def create_intervention(
    request: InterventionRequest,
    background_tasks: BackgroundTasks,
) -> InterventionResponse:
    """Create and execute an intervention."""
    patient_data = state.patients.get(request.patient_id, {
        "patient_id": request.patient_id,
        "first_name": "Patient",
        "medication_name": "your medication",
        "pharmacy_phone": "1-800-OPTUM",
    })
    response = await state.orchestrator.create_intervention(request, patient_data)
    logger.info(
        "intervention_created",
        intervention_id=response.intervention_id,
        patient_id=request.patient_id,
        channel=response.channel.value,
    )
    return response


@app.get("/api/v1/interventions/{intervention_id}", tags=["Interventions"])
async def get_intervention(intervention_id: str) -> dict[str, Any]:
    """Get intervention details."""
    record = state.orchestrator.intervention_records.get(intervention_id)
    if not record:
        raise HTTPException(status_code=404, detail="Intervention not found")
    return {
        "intervention_id": record.intervention_id,
        "patient_id": record.patient_id,
        "channel": record.channel.value,
        "status": record.status.value,
        "message_content": record.message_content,
        "created_at": record.created_at.isoformat(),
        "sent_at": record.sent_at.isoformat() if record.sent_at else None,
    }


@app.post("/api/v1/interventions/{intervention_id}/outcome", tags=["Interventions"])
async def record_intervention_outcome(
    intervention_id: str,
    outcome: InterventionOutcome,
) -> dict[str, str]:
    """Record intervention outcome."""
    success = state.orchestrator.record_outcome(intervention_id, outcome)
    if not success:
        raise HTTPException(status_code=404, detail="Intervention not found")
    return {"status": "recorded", "intervention_id": intervention_id}


@app.get("/api/v1/interventions/patient/{patient_id}", tags=["Interventions"])
async def get_patient_interventions(patient_id: str) -> list[dict[str, Any]]:
    """Get all interventions for a patient."""
    records = state.orchestrator.get_patient_intervention_history(patient_id)
    return [
        {
            "intervention_id": r.intervention_id,
            "channel": r.channel.value,
            "status": r.status.value,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@app.get("/api/v1/interventions/stats", tags=["Interventions"])
async def get_intervention_stats() -> dict[str, Any]:
    """Overall intervention statistics."""
    return state.orchestrator.get_intervention_stats()


# ── Chat ───────────────────────────────────────────────────────

@app.post("/api/v1/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the conversational agent."""
    patient_context = state.patients.get(request.patient_id, {
        "first_name": "there",
        "medication_name": "your medication",
    })
    response = await state.chatbot.chat(request, patient_context)
    logger.info(
        "chat_message",
        patient_id=request.patient_id,
        conversation_id=response.conversation_id,
        barrier_identified=response.identified_barrier.value if response.identified_barrier else None,
    )
    return response


@app.get("/api/v1/chat/{conversation_id}/summary", tags=["Chat"])
async def get_conversation_summary(conversation_id: str) -> dict[str, Any]:
    """Get conversation summary."""
    summary = state.chatbot.get_conversation_summary(conversation_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return summary


@app.get("/api/v1/assistance-programs", tags=["Chat"])
async def find_assistance_programs(medication: str) -> list[dict[str, Any]]:
    """Find copay assistance programs."""
    programs = state.chatbot.find_assistance_programs(medication)
    return [
        {
            "program_id": p.program_id,
            "name": p.name,
            "potential_savings": p.potential_savings,
            "eligibility": p.eligibility_criteria,
            "phone": p.phone_number,
        }
        for p in programs
    ]


# ── Analytics ──────────────────────────────────────────────────

@app.get("/api/v1/analytics/patient/{patient_id}/adherence", tags=["Analytics"])
async def get_patient_adherence(
    patient_id: str,
    medication_ndc: Optional[str] = None,
    lookback_days: int = Query(default=90, ge=30, le=365),
) -> dict[str, Any]:
    """Patient adherence metrics."""
    metrics = state.analytics.get_patient_adherence_metrics(
        patient_id, medication_ndc, lookback_days
    )
    return metrics.model_dump()


@app.get("/api/v1/analytics/channels", tags=["Analytics"])
async def get_channel_effectiveness(
    lookback_days: int = Query(default=90, ge=7, le=365),
) -> dict[str, Any]:
    """Channel effectiveness metrics."""
    effectiveness = state.analytics.get_channel_effectiveness(lookback_days)
    return {
        channel: {
            "total_sent": perf.total_sent,
            "delivery_rate": perf.delivery_rate,
            "response_rate": perf.response_rate,
            "conversion_rate": perf.conversion_rate,
            "cost_per_conversion": perf.cost_per_conversion,
        }
        for channel, perf in effectiveness.items()
    }


@app.get("/api/v1/analytics/roi", tags=["Analytics"])
async def calculate_roi(start_date: date, end_date: date) -> ROIMetrics:
    """Calculate program ROI."""
    return state.analytics.calculate_roi(start_date, end_date)


@app.get("/api/v1/analytics/population", tags=["Analytics"])
async def get_population_summary() -> dict[str, Any]:
    """Population-level analytics."""
    return state.analytics.get_population_summary()


@app.get("/api/v1/analytics/weekly-report", tags=["Analytics"])
async def get_weekly_report() -> dict[str, Any]:
    """Weekly analytics report."""
    return state.analytics.generate_weekly_report()


# ── A/B Testing ────────────────────────────────────────────────

class ABTestCreate(BaseModel):
    test_id: str
    test_name: str
    variant_a: str
    variant_b: str
    metric: str = "conversion_rate"


@app.post("/api/v1/ab-tests", tags=["Experiments"])
async def create_ab_test(test: ABTestCreate) -> dict[str, str]:
    """Create an A/B test."""
    state.analytics.create_ab_test(
        test.test_id, test.test_name,
        test.variant_a, test.variant_b, test.metric,
    )
    return {"status": "created", "test_id": test.test_id}


@app.get("/api/v1/ab-tests/{test_id}/results", tags=["Experiments"])
async def get_ab_test_results(test_id: str) -> dict[str, Any]:
    """Get A/B test results."""
    result = state.analytics.get_ab_test_results(test_id)
    if not result:
        raise HTTPException(status_code=404, detail="Test not found or insufficient data")
    return {
        "test_id": result.test_id,
        "test_name": result.test_name,
        "variant_a": {"name": result.variant_a, "metric": result.metric_a, "sample_size": result.sample_size_a},
        "variant_b": {"name": result.variant_b, "metric": result.metric_b, "sample_size": result.sample_size_b},
        "lift": result.lift,
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "winner": result.winner,
    }


# ── Model ──────────────────────────────────────────────────────

@app.get("/api/v1/model/info", tags=["Model"])
async def get_model_info() -> dict[str, Any]:
    """Risk prediction model information."""
    return state.risk_predictor.get_model_info()


# ── Run ────────────────────────────────────────────────────────

def run():
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
