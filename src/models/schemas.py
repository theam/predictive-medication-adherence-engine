"""
Pydantic schemas for data validation and serialization.
"""
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk level categories for adherence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Gender(str, Enum):
    """Patient gender options."""

    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class PlanType(str, Enum):
    """Insurance plan types."""

    COMMERCIAL = "commercial"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    EXCHANGE = "exchange"


class InterventionChannel(str, Enum):
    """Available intervention channels."""

    SMS = "sms"
    VOICE = "voice"
    EMAIL = "email"
    CHATBOT = "chatbot"
    CARE_MANAGER = "care_manager"
    PUSH_NOTIFICATION = "push_notification"


class InterventionStatus(str, Enum):
    """Status of an intervention."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    RESPONDED = "responded"
    SUCCESSFUL = "successful"
    FAILED = "failed"


class BarrierType(str, Enum):
    """Types of adherence barriers."""

    COST = "cost"
    FORGETFULNESS = "forgetfulness"
    SIDE_EFFECTS = "side_effects"
    COMPLEXITY = "complexity"
    LACK_OF_UNDERSTANDING = "lack_of_understanding"
    ACCESS = "access"
    BELIEFS = "beliefs"
    OTHER = "other"


# ============== Patient & Medication Schemas ==============


class PatientBase(BaseModel):
    """Base patient information."""

    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    plan_type: PlanType
    zip_code: Optional[str] = None


class PatientCreate(PatientBase):
    """Schema for creating a new patient."""

    diagnosis_codes: list[str] = Field(default_factory=list)
    preferred_channel: Optional[InterventionChannel] = None
    preferred_contact_time: Optional[str] = None  # e.g., "morning", "evening"
    phone_number: Optional[str] = None
    email: Optional[str] = None


class Patient(PatientBase):
    """Complete patient schema with computed fields."""

    diagnosis_codes: list[str] = Field(default_factory=list)
    preferred_channel: Optional[InterventionChannel] = None
    preferred_contact_time: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MedicationFill(BaseModel):
    """Schema for a medication fill record."""

    fill_id: str = Field(..., description="Unique fill identifier")
    patient_id: str
    medication_ndc: str = Field(..., description="National Drug Code")
    medication_name: str
    fill_date: date
    days_supply: int = Field(..., ge=1, le=365)
    refill_number: int = Field(..., ge=0)
    copay_amount: float = Field(..., ge=0)
    quantity: int = Field(..., ge=1)
    pharmacy_npi: Optional[str] = None
    prescriber_npi: Optional[str] = None

    @field_validator("medication_ndc")
    @classmethod
    def validate_ndc(cls, v: str) -> str:
        """Validate NDC format (simplified)."""
        cleaned = v.replace("-", "")
        if not cleaned.isdigit() or len(cleaned) not in (10, 11):
            raise ValueError("Invalid NDC format")
        return v


class PatientMedicationProfile(BaseModel):
    """Complete medication profile for a patient."""

    patient: Patient
    medications: list[MedicationFill]
    active_medications_count: int
    total_fills_last_year: int
    average_pdc: float = Field(..., ge=0, le=1, description="Proportion of Days Covered")


# ============== Risk Prediction Schemas ==============


class RiskFactor(BaseModel):
    """Individual risk factor identified by the model."""

    factor_name: str
    impact_score: float = Field(..., description="Impact on risk score")
    description: str
    actionable: bool = True


class RiskPredictionRequest(BaseModel):
    """Request schema for risk prediction."""

    patient_id: str
    medication_ndc: Optional[str] = None
    prediction_horizon_days: int = Field(default=30, ge=7, le=180)


class RiskPredictionResponse(BaseModel):
    """Response schema for risk prediction."""

    patient_id: str
    medication_ndc: Optional[str] = None
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    prediction_horizon_days: int
    top_risk_factors: list[RiskFactor]
    predicted_risk_window_start: date
    predicted_risk_window_end: date
    confidence_score: float = Field(..., ge=0, le=1)
    model_version: str
    prediction_timestamp: datetime


class PopulationRiskSummary(BaseModel):
    """Summary of risk distribution for a population."""

    total_patients: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    average_risk_score: float
    trending_worse: int = Field(..., description="Patients with increasing risk")
    trending_better: int = Field(..., description="Patients with decreasing risk")


# ============== Intervention Schemas ==============


class InterventionRequest(BaseModel):
    """Request to create an intervention."""

    patient_id: str
    risk_prediction_id: Optional[str] = None
    channel: InterventionChannel
    message_template_id: Optional[str] = None
    custom_message: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    priority: int = Field(default=5, ge=1, le=10)


class InterventionResponse(BaseModel):
    """Response for a created/executed intervention."""

    intervention_id: str
    patient_id: str
    channel: InterventionChannel
    status: InterventionStatus
    message_content: str
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    response_received_at: Optional[datetime] = None
    outcome: Optional[str] = None
    created_at: datetime


class InterventionOutcome(BaseModel):
    """Outcome tracking for an intervention."""

    intervention_id: str
    refill_completed: bool
    days_to_refill: Optional[int] = None
    patient_response: Optional[str] = None
    barrier_identified: Optional[BarrierType] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None


# ============== Conversation/Chatbot Schemas ==============


class ConversationMessage(BaseModel):
    """Single message in a conversation."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationContext(BaseModel):
    """Context for a chatbot conversation."""

    patient_id: str
    conversation_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    identified_barriers: list[BarrierType] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    requires_escalation: bool = False
    escalation_type: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for chatbot interaction."""

    patient_id: str
    conversation_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    """Response from chatbot."""

    conversation_id: str
    response: str
    identified_barrier: Optional[BarrierType] = None
    suggested_action: Optional[str] = None
    requires_human_followup: bool = False
    sentiment: Optional[str] = None


# ============== Analytics Schemas ==============


class AdherenceMetrics(BaseModel):
    """Adherence metrics for a patient or population."""

    pdc_current: float = Field(..., ge=0, le=1, description="Current PDC")
    pdc_previous: float = Field(..., ge=0, le=1, description="Previous period PDC")
    pdc_change: float = Field(..., description="Change in PDC")
    refill_rate: float = Field(..., ge=0, le=1)
    gaps_count: int = Field(..., ge=0)
    average_gap_days: Optional[float] = None


class InterventionEffectiveness(BaseModel):
    """Metrics for intervention effectiveness."""

    channel: InterventionChannel
    total_sent: int
    response_rate: float
    success_rate: float
    average_time_to_refill_days: Optional[float]
    cost_per_success: Optional[float]


class ROIMetrics(BaseModel):
    """ROI calculations for the adherence program."""

    period_start: date
    period_end: date
    patients_intervened: int
    successful_interventions: int
    estimated_hospitalizations_avoided: int
    estimated_cost_savings: float
    program_cost: float
    roi_percentage: float
