"""
Intervention Orchestrator Service.

Coordinates the end-to-end intervention workflow:
1. Receives risk predictions
2. Selects optimal channel and timing
3. Generates personalized messages
4. Dispatches interventions
5. Tracks outcomes
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import structlog

from ..models.schemas import (
    BarrierType,
    InterventionChannel,
    InterventionOutcome,
    InterventionRequest,
    InterventionResponse,
    InterventionStatus,
    RiskLevel,
)
from ..models.risk_predictor import RiskPrediction
from .channel_selector import ChannelRecommendation, ChannelSelector
from .message_templates import MessageTemplateEngine, PersonalizedMessage

logger = structlog.get_logger()


@dataclass
class InterventionRule:
    """Rule for triggering interventions."""

    rule_id: str
    name: str
    risk_level_min: RiskLevel
    condition: Callable[[dict[str, Any]], bool]
    priority: int
    cooldown_hours: int = 24
    max_per_day: int = 3


@dataclass
class ScheduledIntervention:
    """An intervention scheduled for future delivery."""

    intervention_id: str
    patient_id: str
    channel: InterventionChannel
    message: PersonalizedMessage
    scheduled_time: datetime
    priority: int
    status: InterventionStatus = InterventionStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionRecord:
    """Record of a completed or in-progress intervention."""

    intervention_id: str
    patient_id: str
    risk_prediction_id: Optional[str]
    channel: InterventionChannel
    message_content: str
    status: InterventionStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    response_received_at: Optional[datetime] = None
    outcome: Optional[InterventionOutcome] = None


class InterventionOrchestrator:
    """
    Orchestrates medication adherence interventions.

    Features:
    - Rule-based intervention triggering
    - Channel and timing optimization
    - Rate limiting and cooldowns
    - Multi-channel fallback
    - Outcome tracking
    """

    def __init__(
        self,
        channel_selector: Optional[ChannelSelector] = None,
        template_engine: Optional[MessageTemplateEngine] = None,
        max_interventions_per_day: int = 3,
        default_cooldown_hours: int = 24,
    ):
        self.channel_selector = channel_selector or ChannelSelector()
        self.template_engine = template_engine or MessageTemplateEngine()
        self.max_interventions_per_day = max_interventions_per_day
        self.default_cooldown_hours = default_cooldown_hours

        # Storage (in production, use database)
        self.intervention_records: dict[str, InterventionRecord] = {}
        self.scheduled_queue: list[ScheduledIntervention] = []
        self.patient_intervention_counts: dict[str, list[datetime]] = {}

        # Channel handlers (pluggable)
        self.channel_handlers: dict[InterventionChannel, Callable] = {}

        # Rules
        self.rules = self._initialize_rules()

        self._log = logger.bind(service="intervention_orchestrator")

    def _initialize_rules(self) -> list[InterventionRule]:
        """Initialize intervention rules."""
        return [
            InterventionRule(
                rule_id="high_risk_immediate",
                name="High Risk Immediate Intervention",
                risk_level_min=RiskLevel.HIGH,
                condition=lambda ctx: ctx.get("risk_score", 0) >= 70,
                priority=1,
                cooldown_hours=12,
                max_per_day=2,
            ),
            InterventionRule(
                rule_id="overdue_refill",
                name="Overdue Refill Reminder",
                risk_level_min=RiskLevel.MEDIUM,
                condition=lambda ctx: ctx.get("is_overdue", False),
                priority=2,
                cooldown_hours=48,
                max_per_day=1,
            ),
            InterventionRule(
                rule_id="upcoming_gap",
                name="Predicted Gap Prevention",
                risk_level_min=RiskLevel.MEDIUM,
                condition=lambda ctx: ctx.get("days_until_gap", 999) <= 7,
                priority=3,
                cooldown_hours=72,
                max_per_day=1,
            ),
            InterventionRule(
                rule_id="cost_barrier",
                name="Cost Barrier Assistance",
                risk_level_min=RiskLevel.LOW,
                condition=lambda ctx: ctx.get("barrier") == BarrierType.COST,
                priority=4,
                cooldown_hours=168,  # 1 week
                max_per_day=1,
            ),
        ]

    def register_channel_handler(
        self,
        channel: InterventionChannel,
        handler: Callable[[str, PersonalizedMessage], bool],
    ) -> None:
        """
        Register a handler for a channel.

        Handler should accept (patient_id, message) and return success boolean.
        """
        self.channel_handlers[channel] = handler
        self._log.info("registered_channel_handler", channel=channel.value)

    async def process_risk_prediction(
        self,
        prediction: RiskPrediction,
        patient_data: dict[str, Any],
        force: bool = False,
    ) -> Optional[InterventionResponse]:
        """
        Process a risk prediction and potentially trigger intervention.

        Args:
            prediction: Risk prediction result
            patient_data: Patient profile data
            force: Bypass rate limiting

        Returns:
            InterventionResponse if intervention triggered, None otherwise
        """
        patient_id = prediction.patient_id

        self._log.info(
            "processing_prediction",
            patient_id=patient_id,
            risk_score=prediction.risk_score,
            risk_level=prediction.risk_level.value,
        )

        # Build context for rule evaluation
        context = {
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level,
            "is_overdue": any(
                f.factor_name == "is_overdue" for f in prediction.top_factors
            ),
            "days_until_gap": patient_data.get("days_until_gap", 999),
            "barrier": patient_data.get("identified_barrier"),
            **patient_data,
        }

        # Check rate limits
        if not force and not self._check_rate_limit(patient_id):
            self._log.info("rate_limited", patient_id=patient_id)
            return None

        # Find applicable rule
        rule = self._find_applicable_rule(prediction.risk_level, context)
        if not rule:
            self._log.info("no_applicable_rule", patient_id=patient_id)
            return None

        # Check rule-specific cooldown
        if not force and not self._check_rule_cooldown(patient_id, rule):
            self._log.info(
                "rule_cooldown_active",
                patient_id=patient_id,
                rule_id=rule.rule_id,
            )
            return None

        # Select channel
        channel_rec = self.channel_selector.select_channel(
            patient_id=patient_id,
            patient_data=patient_data,
            risk_level=prediction.risk_level,
        )

        # Generate message
        message = self.template_engine.generate_adaptive_message(
            channel=channel_rec.primary_channel,
            patient_context={
                "first_name": patient_data.get("first_name", "there"),
                "medication_name": patient_data.get("medication_name", "your medication"),
                "pharmacy_phone": patient_data.get("pharmacy_phone", "1-800-OPTUM"),
                "days_overdue": patient_data.get("days_overdue", 0),
                "copay_amount": patient_data.get("copay_amount", "N/A"),
                **patient_data,
            },
            risk_level=prediction.risk_level,
            barrier=patient_data.get("identified_barrier"),
        )

        # Execute intervention
        response = await self._execute_intervention(
            patient_id=patient_id,
            channel=channel_rec.primary_channel,
            message=message,
            risk_prediction_id=None,  # Would be prediction ID in production
            rule=rule,
            backup_channel=channel_rec.backup_channel,
        )

        return response

    async def create_intervention(
        self,
        request: InterventionRequest,
        patient_data: dict[str, Any],
    ) -> InterventionResponse:
        """
        Create and execute an intervention from explicit request.

        Args:
            request: Intervention request
            patient_data: Patient profile data

        Returns:
            InterventionResponse
        """
        # Generate message
        if request.custom_message:
            message = PersonalizedMessage(
                template_id="custom",
                channel=request.channel,
                subject=None,
                body=request.custom_message,
                cta=None,
                metadata={"custom": True},
            )
        elif request.message_template_id:
            message = self.template_engine.generate_message(
                template_id=request.message_template_id,
                context={
                    "first_name": patient_data.get("first_name", "there"),
                    "medication_name": patient_data.get("medication_name", "your medication"),
                    "pharmacy_phone": patient_data.get("pharmacy_phone", "1-800-OPTUM"),
                    **patient_data,
                },
            )
        else:
            # Use adaptive message
            message = self.template_engine.generate_adaptive_message(
                channel=request.channel,
                patient_context=patient_data,
                risk_level=RiskLevel.MEDIUM,  # Default
            )

        # Execute
        response = await self._execute_intervention(
            patient_id=request.patient_id,
            channel=request.channel,
            message=message,
            risk_prediction_id=request.risk_prediction_id,
        )

        return response

    async def _execute_intervention(
        self,
        patient_id: str,
        channel: InterventionChannel,
        message: PersonalizedMessage,
        risk_prediction_id: Optional[str] = None,
        rule: Optional[InterventionRule] = None,
        backup_channel: Optional[InterventionChannel] = None,
    ) -> InterventionResponse:
        """Execute an intervention through the appropriate channel."""
        intervention_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Create record
        record = InterventionRecord(
            intervention_id=intervention_id,
            patient_id=patient_id,
            risk_prediction_id=risk_prediction_id,
            channel=channel,
            message_content=message.body,
            status=InterventionStatus.PENDING,
            created_at=now,
        )

        self._log.info(
            "executing_intervention",
            intervention_id=intervention_id,
            patient_id=patient_id,
            channel=channel.value,
        )

        # Try to send
        success = await self._send_via_channel(patient_id, channel, message)

        if success:
            record.status = InterventionStatus.SENT
            record.sent_at = datetime.utcnow()
            self._record_intervention(patient_id)
        elif backup_channel:
            # Try backup channel
            self._log.info(
                "trying_backup_channel",
                intervention_id=intervention_id,
                backup_channel=backup_channel.value,
            )
            success = await self._send_via_channel(patient_id, backup_channel, message)
            if success:
                record.channel = backup_channel
                record.status = InterventionStatus.SENT
                record.sent_at = datetime.utcnow()
                self._record_intervention(patient_id)
            else:
                record.status = InterventionStatus.FAILED
        else:
            record.status = InterventionStatus.FAILED

        # Store record
        self.intervention_records[intervention_id] = record

        return InterventionResponse(
            intervention_id=intervention_id,
            patient_id=patient_id,
            channel=record.channel,
            status=record.status,
            message_content=message.body,
            sent_at=record.sent_at,
            created_at=record.created_at,
        )

    async def _send_via_channel(
        self,
        patient_id: str,
        channel: InterventionChannel,
        message: PersonalizedMessage,
    ) -> bool:
        """Send message through specific channel."""
        handler = self.channel_handlers.get(channel)

        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(patient_id, message)
                else:
                    return handler(patient_id, message)
            except Exception as e:
                self._log.error(
                    "channel_handler_error",
                    channel=channel.value,
                    error=str(e),
                )
                return False
        else:
            # No handler registered - simulate success for demo
            self._log.info(
                "simulated_send",
                channel=channel.value,
                patient_id=patient_id,
            )
            return True

    def record_outcome(
        self,
        intervention_id: str,
        outcome: InterventionOutcome,
    ) -> bool:
        """Record the outcome of an intervention."""
        record = self.intervention_records.get(intervention_id)
        if not record:
            return False

        record.outcome = outcome

        if outcome.refill_completed:
            record.status = InterventionStatus.SUCCESSFUL
        elif outcome.patient_response:
            record.status = InterventionStatus.RESPONDED
            record.response_received_at = datetime.utcnow()

        # Update channel selector with engagement data
        self.channel_selector.record_engagement(
            patient_id=record.patient_id,
            channel=record.channel,
            responded=outcome.patient_response is not None,
            response_time_hours=outcome.days_to_refill * 24 if outcome.days_to_refill else None,
        )

        self._log.info(
            "outcome_recorded",
            intervention_id=intervention_id,
            successful=outcome.refill_completed,
        )

        return True

    def _find_applicable_rule(
        self, risk_level: RiskLevel, context: dict[str, Any]
    ) -> Optional[InterventionRule]:
        """Find the highest priority applicable rule."""
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
        patient_risk_value = risk_order.get(risk_level, 0)

        applicable = []
        for rule in self.rules:
            rule_min_value = risk_order.get(rule.risk_level_min, 0)
            if patient_risk_value >= rule_min_value and rule.condition(context):
                applicable.append(rule)

        if not applicable:
            return None

        # Return highest priority (lowest number)
        applicable.sort(key=lambda r: r.priority)
        return applicable[0]

    def _check_rate_limit(self, patient_id: str) -> bool:
        """Check if patient is within rate limits."""
        counts = self.patient_intervention_counts.get(patient_id, [])
        today = datetime.utcnow().date()

        # Count today's interventions
        today_count = sum(1 for ts in counts if ts.date() == today)
        return today_count < self.max_interventions_per_day

    def _check_rule_cooldown(self, patient_id: str, rule: InterventionRule) -> bool:
        """Check if rule cooldown has passed."""
        counts = self.patient_intervention_counts.get(patient_id, [])
        if not counts:
            return True

        cutoff = datetime.utcnow() - timedelta(hours=rule.cooldown_hours)
        recent = [ts for ts in counts if ts > cutoff]
        return len(recent) == 0

    def _record_intervention(self, patient_id: str) -> None:
        """Record that an intervention was sent."""
        if patient_id not in self.patient_intervention_counts:
            self.patient_intervention_counts[patient_id] = []

        self.patient_intervention_counts[patient_id].append(datetime.utcnow())

        # Keep only last 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.patient_intervention_counts[patient_id] = [
            ts
            for ts in self.patient_intervention_counts[patient_id]
            if ts > cutoff
        ]

    def get_patient_intervention_history(
        self, patient_id: str
    ) -> list[InterventionRecord]:
        """Get intervention history for a patient."""
        return [
            record
            for record in self.intervention_records.values()
            if record.patient_id == patient_id
        ]

    def get_intervention_stats(self) -> dict[str, Any]:
        """Get overall intervention statistics."""
        total = len(self.intervention_records)
        if total == 0:
            return {"total": 0}

        by_status = {}
        by_channel = {}

        for record in self.intervention_records.values():
            status = record.status.value
            channel = record.channel.value

            by_status[status] = by_status.get(status, 0) + 1
            by_channel[channel] = by_channel.get(channel, 0) + 1

        successful = by_status.get(InterventionStatus.SUCCESSFUL.value, 0)

        return {
            "total": total,
            "by_status": by_status,
            "by_channel": by_channel,
            "success_rate": round(successful / total, 3) if total > 0 else 0,
        }
