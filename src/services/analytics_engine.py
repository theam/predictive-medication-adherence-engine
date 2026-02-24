"""
Closed-Loop Analytics Engine.

Tracks intervention effectiveness, measures ROI, and provides
insights for continuous optimization of the adherence program.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional
from collections import defaultdict

import numpy as np

from ..models.schemas import (
    AdherenceMetrics,
    InterventionChannel,
    InterventionEffectiveness,
    InterventionStatus,
    ROIMetrics,
    RiskLevel,
)


@dataclass
class PatientAdherenceRecord:
    """Track patient adherence over time."""

    patient_id: str
    medication_ndc: str
    pdc_history: list[tuple[date, float]] = field(default_factory=list)
    interventions_received: list[dict] = field(default_factory=list)
    refills: list[dict] = field(default_factory=list)


@dataclass
class ABTestResult:
    """Results from an A/B test."""

    test_id: str
    test_name: str
    variant_a: str
    variant_b: str
    sample_size_a: int
    sample_size_b: int
    metric_a: float
    metric_b: float
    lift: float
    p_value: float
    is_significant: bool
    winner: Optional[str]


@dataclass
class ChannelPerformance:
    """Detailed channel performance metrics."""

    channel: InterventionChannel
    total_sent: int
    delivered: int
    opened: int
    responded: int
    converted: int  # Led to refill
    delivery_rate: float
    open_rate: float
    response_rate: float
    conversion_rate: float
    avg_time_to_response_hours: float
    avg_time_to_refill_days: float
    cost_per_intervention: float
    cost_per_conversion: float


class AnalyticsEngine:
    """
    Analytics engine for measuring and optimizing adherence interventions.

    Features:
    - PDC tracking and trending
    - Intervention effectiveness measurement
    - A/B test management and analysis
    - ROI calculation
    - Population-level insights
    """

    # Cost assumptions for ROI calculations
    COST_PER_HOSPITALIZATION = 15000
    COST_PER_ER_VISIT = 2500
    HOSPITALIZATION_REDUCTION_RATE = 0.15  # 15% reduction per adherent patient
    INTERVENTION_COSTS = {
        InterventionChannel.SMS: 0.05,
        InterventionChannel.EMAIL: 0.02,
        InterventionChannel.VOICE: 1.50,
        InterventionChannel.PUSH_NOTIFICATION: 0.01,
        InterventionChannel.CHATBOT: 0.25,
        InterventionChannel.CARE_MANAGER: 25.00,
    }

    def __init__(self):
        # Storage (use database in production)
        self.patient_records: dict[str, PatientAdherenceRecord] = {}
        self.intervention_outcomes: list[dict] = []
        self.ab_tests: dict[str, dict] = {}

    def record_pdc(
        self,
        patient_id: str,
        medication_ndc: str,
        pdc: float,
        as_of_date: Optional[date] = None,
    ) -> None:
        """Record PDC measurement for a patient."""
        if as_of_date is None:
            as_of_date = date.today()

        key = f"{patient_id}:{medication_ndc}"

        if key not in self.patient_records:
            self.patient_records[key] = PatientAdherenceRecord(
                patient_id=patient_id,
                medication_ndc=medication_ndc,
            )

        self.patient_records[key].pdc_history.append((as_of_date, pdc))

    def record_intervention_outcome(
        self,
        intervention_id: str,
        patient_id: str,
        channel: InterventionChannel,
        status: InterventionStatus,
        sent_at: datetime,
        delivered_at: Optional[datetime] = None,
        opened_at: Optional[datetime] = None,
        responded_at: Optional[datetime] = None,
        refill_completed: bool = False,
        refill_date: Optional[date] = None,
        ab_test_id: Optional[str] = None,
        ab_variant: Optional[str] = None,
    ) -> None:
        """Record the outcome of an intervention."""
        outcome = {
            "intervention_id": intervention_id,
            "patient_id": patient_id,
            "channel": channel.value,
            "status": status.value,
            "sent_at": sent_at.isoformat(),
            "delivered_at": delivered_at.isoformat() if delivered_at else None,
            "opened_at": opened_at.isoformat() if opened_at else None,
            "responded_at": responded_at.isoformat() if responded_at else None,
            "refill_completed": refill_completed,
            "refill_date": refill_date.isoformat() if refill_date else None,
            "ab_test_id": ab_test_id,
            "ab_variant": ab_variant,
            "recorded_at": datetime.utcnow().isoformat(),
        }

        self.intervention_outcomes.append(outcome)

    def get_patient_adherence_metrics(
        self,
        patient_id: str,
        medication_ndc: Optional[str] = None,
        lookback_days: int = 90,
    ) -> AdherenceMetrics:
        """Get adherence metrics for a patient."""
        # Find matching records
        records = [
            r for key, r in self.patient_records.items()
            if r.patient_id == patient_id
            and (medication_ndc is None or r.medication_ndc == medication_ndc)
        ]

        if not records:
            return AdherenceMetrics(
                pdc_current=0.0,
                pdc_previous=0.0,
                pdc_change=0.0,
                refill_rate=0.0,
                gaps_count=0,
            )

        # Aggregate PDC from all medications
        cutoff = date.today() - timedelta(days=lookback_days)
        previous_cutoff = cutoff - timedelta(days=lookback_days)

        current_pdcs = []
        previous_pdcs = []

        for record in records:
            for pdc_date, pdc in record.pdc_history:
                if pdc_date >= cutoff:
                    current_pdcs.append(pdc)
                elif pdc_date >= previous_cutoff:
                    previous_pdcs.append(pdc)

        pdc_current = np.mean(current_pdcs) if current_pdcs else 0.0
        pdc_previous = np.mean(previous_pdcs) if previous_pdcs else 0.0
        pdc_change = pdc_current - pdc_previous

        # Count gaps (simplified)
        gaps = sum(1 for record in records for _, pdc in record.pdc_history if pdc < 0.8)

        return AdherenceMetrics(
            pdc_current=round(pdc_current, 3),
            pdc_previous=round(pdc_previous, 3),
            pdc_change=round(pdc_change, 3),
            refill_rate=round(pdc_current, 3),  # Simplified
            gaps_count=gaps,
        )

    def get_channel_effectiveness(
        self,
        lookback_days: int = 90,
    ) -> dict[str, ChannelPerformance]:
        """Calculate effectiveness metrics for each channel."""
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        # Filter recent outcomes
        recent = [
            o for o in self.intervention_outcomes
            if datetime.fromisoformat(o["sent_at"]) >= cutoff
        ]

        # Group by channel
        by_channel = defaultdict(list)
        for outcome in recent:
            by_channel[outcome["channel"]].append(outcome)

        results = {}

        for channel_str, outcomes in by_channel.items():
            channel = InterventionChannel(channel_str)
            total = len(outcomes)

            if total == 0:
                continue

            delivered = sum(1 for o in outcomes if o.get("delivered_at"))
            opened = sum(1 for o in outcomes if o.get("opened_at"))
            responded = sum(1 for o in outcomes if o.get("responded_at"))
            converted = sum(1 for o in outcomes if o.get("refill_completed"))

            # Calculate response times
            response_times = []
            refill_times = []

            for o in outcomes:
                if o.get("responded_at") and o.get("sent_at"):
                    sent = datetime.fromisoformat(o["sent_at"])
                    responded_at = datetime.fromisoformat(o["responded_at"])
                    response_times.append((responded_at - sent).total_seconds() / 3600)

                if o.get("refill_date") and o.get("sent_at"):
                    sent = datetime.fromisoformat(o["sent_at"])
                    refill = date.fromisoformat(o["refill_date"])
                    refill_times.append((refill - sent.date()).days)

            cost_per = self.INTERVENTION_COSTS.get(channel, 0.10)
            total_cost = total * cost_per

            results[channel_str] = ChannelPerformance(
                channel=channel,
                total_sent=total,
                delivered=delivered,
                opened=opened,
                responded=responded,
                converted=converted,
                delivery_rate=round(delivered / total, 3) if total > 0 else 0,
                open_rate=round(opened / delivered, 3) if delivered > 0 else 0,
                response_rate=round(responded / total, 3) if total > 0 else 0,
                conversion_rate=round(converted / total, 3) if total > 0 else 0,
                avg_time_to_response_hours=round(np.mean(response_times), 1) if response_times else 0,
                avg_time_to_refill_days=round(np.mean(refill_times), 1) if refill_times else 0,
                cost_per_intervention=cost_per,
                cost_per_conversion=round(total_cost / converted, 2) if converted > 0 else 0,
            )

        return results

    def calculate_roi(
        self,
        period_start: date,
        period_end: date,
    ) -> ROIMetrics:
        """Calculate ROI for the adherence program."""
        # Get interventions in period
        interventions = [
            o for o in self.intervention_outcomes
            if period_start <= date.fromisoformat(o["sent_at"][:10]) <= period_end
        ]

        patients_intervened = len(set(o["patient_id"] for o in interventions))
        successful = sum(1 for o in interventions if o.get("refill_completed"))

        # Calculate intervention costs
        program_cost = sum(
            self.INTERVENTION_COSTS.get(InterventionChannel(o["channel"]), 0.10)
            for o in interventions
        )

        # Estimate savings
        # Assumption: Each successful intervention prevents some probability of hospitalization
        estimated_hospitalizations_avoided = int(
            successful * self.HOSPITALIZATION_REDUCTION_RATE
        )

        estimated_cost_savings = (
            estimated_hospitalizations_avoided * self.COST_PER_HOSPITALIZATION
            + (successful * 0.05) * self.COST_PER_ER_VISIT  # 5% ER visit reduction
        )

        # Calculate ROI
        if program_cost > 0:
            roi_percentage = ((estimated_cost_savings - program_cost) / program_cost) * 100
        else:
            roi_percentage = 0.0

        return ROIMetrics(
            period_start=period_start,
            period_end=period_end,
            patients_intervened=patients_intervened,
            successful_interventions=successful,
            estimated_hospitalizations_avoided=estimated_hospitalizations_avoided,
            estimated_cost_savings=round(estimated_cost_savings, 2),
            program_cost=round(program_cost, 2),
            roi_percentage=round(roi_percentage, 1),
        )

    def create_ab_test(
        self,
        test_id: str,
        test_name: str,
        variant_a: str,
        variant_b: str,
        metric: str = "conversion_rate",
    ) -> None:
        """Create a new A/B test."""
        self.ab_tests[test_id] = {
            "test_id": test_id,
            "test_name": test_name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metric": metric,
            "created_at": datetime.utcnow().isoformat(),
            "status": "running",
        }

    def get_ab_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze results of an A/B test."""
        test = self.ab_tests.get(test_id)
        if not test:
            return None

        # Get outcomes for this test
        test_outcomes = [
            o for o in self.intervention_outcomes
            if o.get("ab_test_id") == test_id
        ]

        variant_a_outcomes = [o for o in test_outcomes if o.get("ab_variant") == "a"]
        variant_b_outcomes = [o for o in test_outcomes if o.get("ab_variant") == "b"]

        n_a = len(variant_a_outcomes)
        n_b = len(variant_b_outcomes)

        if n_a < 10 or n_b < 10:
            return None  # Not enough data

        # Calculate metrics
        metric = test.get("metric", "conversion_rate")

        if metric == "conversion_rate":
            conversions_a = sum(1 for o in variant_a_outcomes if o.get("refill_completed"))
            conversions_b = sum(1 for o in variant_b_outcomes if o.get("refill_completed"))
            metric_a = conversions_a / n_a
            metric_b = conversions_b / n_b
        elif metric == "response_rate":
            responses_a = sum(1 for o in variant_a_outcomes if o.get("responded_at"))
            responses_b = sum(1 for o in variant_b_outcomes if o.get("responded_at"))
            metric_a = responses_a / n_a
            metric_b = responses_b / n_b
        else:
            metric_a = 0
            metric_b = 0

        # Calculate lift
        lift = ((metric_b - metric_a) / metric_a * 100) if metric_a > 0 else 0

        # Simple z-test for proportions
        p_pooled = (metric_a * n_a + metric_b * n_b) / (n_a + n_b)
        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
            z_score = abs(metric_b - metric_a) / se if se > 0 else 0
            # Approximate p-value (two-tailed)
            p_value = 2 * (1 - min(0.9999, 0.5 + z_score * 0.15))  # Simplified
        else:
            p_value = 1.0

        is_significant = p_value < 0.05

        # Determine winner
        winner = None
        if is_significant:
            if metric_b > metric_a:
                winner = test["variant_b"]
            else:
                winner = test["variant_a"]

        return ABTestResult(
            test_id=test_id,
            test_name=test["test_name"],
            variant_a=test["variant_a"],
            variant_b=test["variant_b"],
            sample_size_a=n_a,
            sample_size_b=n_b,
            metric_a=round(metric_a, 4),
            metric_b=round(metric_b, 4),
            lift=round(lift, 2),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            winner=winner,
        )

    def get_population_summary(
        self,
        risk_distribution: Optional[dict[RiskLevel, int]] = None,
    ) -> dict[str, Any]:
        """Get summary statistics for the patient population."""
        total_patients = len(set(
            r.patient_id for r in self.patient_records.values()
        ))

        # Calculate average PDC
        all_pdcs = []
        for record in self.patient_records.values():
            if record.pdc_history:
                latest_pdc = record.pdc_history[-1][1]
                all_pdcs.append(latest_pdc)

        avg_pdc = np.mean(all_pdcs) if all_pdcs else 0

        # Count patients by adherence level
        adherent_count = sum(1 for pdc in all_pdcs if pdc >= 0.8)
        partially_adherent = sum(1 for pdc in all_pdcs if 0.5 <= pdc < 0.8)
        non_adherent = sum(1 for pdc in all_pdcs if pdc < 0.5)

        # Trending
        improving = 0
        worsening = 0
        stable = 0

        for record in self.patient_records.values():
            if len(record.pdc_history) >= 2:
                recent = record.pdc_history[-1][1]
                previous = record.pdc_history[-2][1]
                if recent > previous + 0.05:
                    improving += 1
                elif recent < previous - 0.05:
                    worsening += 1
                else:
                    stable += 1

        return {
            "total_patients": total_patients,
            "average_pdc": round(avg_pdc, 3),
            "adherent_count": adherent_count,
            "partially_adherent_count": partially_adherent,
            "non_adherent_count": non_adherent,
            "adherent_percentage": round(adherent_count / max(total_patients, 1) * 100, 1),
            "trending_improving": improving,
            "trending_worsening": worsening,
            "trending_stable": stable,
            "risk_distribution": {
                k.value: v for k, v in (risk_distribution or {}).items()
            },
        }

    def generate_weekly_report(self) -> dict[str, Any]:
        """Generate a weekly analytics report."""
        today = date.today()
        week_ago = today - timedelta(days=7)
        two_weeks_ago = today - timedelta(days=14)

        # Current week metrics
        current_roi = self.calculate_roi(week_ago, today)
        previous_roi = self.calculate_roi(two_weeks_ago, week_ago)

        # Channel effectiveness
        channel_effectiveness = self.get_channel_effectiveness(lookback_days=7)

        # Active A/B tests
        active_tests = []
        for test_id in self.ab_tests:
            result = self.get_ab_test_results(test_id)
            if result:
                active_tests.append({
                    "test_name": result.test_name,
                    "lift": result.lift,
                    "is_significant": result.is_significant,
                    "winner": result.winner,
                    "sample_size": result.sample_size_a + result.sample_size_b,
                })

        return {
            "report_date": today.isoformat(),
            "period": {"start": week_ago.isoformat(), "end": today.isoformat()},
            "summary": {
                "patients_intervened": current_roi.patients_intervened,
                "successful_interventions": current_roi.successful_interventions,
                "success_rate": round(
                    current_roi.successful_interventions / max(current_roi.patients_intervened, 1),
                    3,
                ),
                "estimated_savings": current_roi.estimated_cost_savings,
                "roi_percentage": current_roi.roi_percentage,
            },
            "week_over_week": {
                "interventions_change": (
                    current_roi.patients_intervened - previous_roi.patients_intervened
                ),
                "success_rate_change": round(
                    (current_roi.successful_interventions / max(current_roi.patients_intervened, 1))
                    - (previous_roi.successful_interventions / max(previous_roi.patients_intervened, 1)),
                    3,
                ),
            },
            "channel_performance": {
                channel: {
                    "total_sent": perf.total_sent,
                    "conversion_rate": perf.conversion_rate,
                    "cost_per_conversion": perf.cost_per_conversion,
                }
                for channel, perf in channel_effectiveness.items()
            },
            "ab_tests": active_tests,
        }
