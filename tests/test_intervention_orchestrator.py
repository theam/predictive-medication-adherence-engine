"""
Tests for the intervention orchestrator.
"""
from datetime import datetime

import pytest

from src.models.schemas import (
    InterventionChannel,
    InterventionRequest,
    InterventionStatus,
    RiskLevel,
)
from src.models.risk_predictor import RiskPrediction, RiskFactor
from src.services.intervention_orchestrator import InterventionOrchestrator


@pytest.fixture
def orchestrator():
    """Create an orchestrator instance."""
    return InterventionOrchestrator(
        max_interventions_per_day=3,
        default_cooldown_hours=24,
    )


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return {
        "patient_id": "P001",
        "first_name": "John",
        "medication_name": "Metformin 500mg",
        "phone_number": "+15551234567",
        "email": "john@example.com",
        "pharmacy_phone": "1-800-555-0100",
    }


@pytest.fixture
def high_risk_prediction():
    """Sample high-risk prediction."""
    return RiskPrediction(
        patient_id="P001",
        medication_ndc="00093-7212-01",
        risk_score=85.0,
        risk_level=RiskLevel.HIGH,
        top_factors=[
            RiskFactor(
                factor_name="pdc_90_days",
                impact_score=0.35,
                description="Medication coverage in last 90 days",
                actionable=True,
            ),
            RiskFactor(
                factor_name="gap_count",
                impact_score=0.25,
                description="Number of gaps in medication coverage",
                actionable=True,
            ),
        ],
        prediction_horizon_days=30,
        confidence=0.85,
        model_version="1.0.0",
    )


class TestInterventionOrchestrator:
    """Tests for InterventionOrchestrator."""

    @pytest.mark.asyncio
    async def test_create_intervention_returns_response(
        self, orchestrator, sample_patient_data
    ):
        """Test that create_intervention returns a response."""
        request = InterventionRequest(
            patient_id="P001",
            channel=InterventionChannel.SMS,
        )

        response = await orchestrator.create_intervention(request, sample_patient_data)

        assert response.intervention_id is not None
        assert response.patient_id == "P001"
        assert response.channel == InterventionChannel.SMS
        assert response.status in [InterventionStatus.SENT, InterventionStatus.PENDING]

    @pytest.mark.asyncio
    async def test_process_risk_prediction(
        self, orchestrator, sample_patient_data, high_risk_prediction
    ):
        """Test processing a high-risk prediction triggers intervention."""
        response = await orchestrator.process_risk_prediction(
            high_risk_prediction,
            sample_patient_data,
            force=True,  # Bypass rate limiting for test
        )

        # High risk should trigger intervention
        assert response is not None
        assert response.patient_id == "P001"

    @pytest.mark.asyncio
    async def test_rate_limiting(
        self, orchestrator, sample_patient_data, high_risk_prediction
    ):
        """Test that rate limiting works."""
        # Send max interventions
        for _ in range(orchestrator.max_interventions_per_day):
            await orchestrator.process_risk_prediction(
                high_risk_prediction,
                sample_patient_data,
                force=True,
            )
            orchestrator._record_intervention("P001")

        # Next one should be rate limited
        response = await orchestrator.process_risk_prediction(
            high_risk_prediction,
            sample_patient_data,
            force=False,  # Enable rate limiting
        )

        assert response is None

    @pytest.mark.asyncio
    async def test_custom_message(self, orchestrator, sample_patient_data):
        """Test intervention with custom message."""
        request = InterventionRequest(
            patient_id="P001",
            channel=InterventionChannel.SMS,
            custom_message="Custom test message",
        )

        response = await orchestrator.create_intervention(request, sample_patient_data)

        assert response.message_content == "Custom test message"

    def test_get_intervention_stats(self, orchestrator):
        """Test stats retrieval."""
        stats = orchestrator.get_intervention_stats()

        assert "total" in stats
        assert stats["total"] >= 0

    def test_intervention_history(self, orchestrator):
        """Test getting patient intervention history."""
        history = orchestrator.get_patient_intervention_history("P001")

        assert isinstance(history, list)


class TestChannelSelector:
    """Tests for channel selection."""

    def test_channel_selection_respects_preference(self):
        """Test that preferred channel is prioritized."""
        from src.services.channel_selector import ChannelSelector

        selector = ChannelSelector()

        patient_data = {
            "patient_id": "P001",
            "phone_number": "+15551234567",
            "email": "test@example.com",
            "preferred_channel": "email",
        }

        recommendation = selector.select_channel(
            patient_id="P001",
            patient_data=patient_data,
            risk_level=RiskLevel.MEDIUM,
        )

        # Preferred channel should be weighted heavily
        email_score = next(
            (s for s in recommendation.scores if s.channel == InterventionChannel.EMAIL),
            None,
        )
        assert email_score is not None
        assert "preferred" in email_score.reason.lower()

    def test_channel_selection_high_risk(self):
        """Test that high risk patients get appropriate channels."""
        from src.services.channel_selector import ChannelSelector

        selector = ChannelSelector()

        patient_data = {
            "patient_id": "P001",
            "phone_number": "+15551234567",
            "email": "test@example.com",
        }

        recommendation = selector.select_channel(
            patient_id="P001",
            patient_data=patient_data,
            risk_level=RiskLevel.HIGH,
        )

        # High risk should prioritize care_manager or voice
        assert recommendation.primary_channel in [
            InterventionChannel.CARE_MANAGER,
            InterventionChannel.VOICE,
            InterventionChannel.SMS,
        ]


class TestMessageTemplates:
    """Tests for message templates."""

    def test_template_generation(self):
        """Test message template generation."""
        from src.services.message_templates import (
            MessageTemplateEngine,
            MessagePurpose,
        )

        engine = MessageTemplateEngine()

        message = engine.generate_message(
            template_id="sms_refill_reminder_friendly",
            context={
                "first_name": "John",
                "medication_name": "Metformin",
                "pharmacy_phone": "1-800-555-0100",
            },
        )

        assert "John" in message.body
        assert "Metformin" in message.body
        assert message.channel == InterventionChannel.SMS

    def test_adaptive_message_generation(self):
        """Test adaptive message generation based on context."""
        from src.services.message_templates import MessageTemplateEngine
        from src.models.schemas import BarrierType

        engine = MessageTemplateEngine()

        # Cost barrier should generate cost assistance message
        message = engine.generate_adaptive_message(
            channel=InterventionChannel.SMS,
            patient_context={
                "first_name": "Jane",
                "medication_name": "Atorvastatin",
            },
            risk_level=RiskLevel.MEDIUM,
            barrier=BarrierType.COST,
        )

        assert "sav" in message.body.lower() or "cost" in message.body.lower()

    def test_list_templates(self):
        """Test listing available templates."""
        from src.services.message_templates import MessageTemplateEngine

        engine = MessageTemplateEngine()
        templates = engine.list_templates()

        assert len(templates) > 0
        assert all("template_id" in t for t in templates)
