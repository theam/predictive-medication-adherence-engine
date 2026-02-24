"""
Channel Selection Service.

Uses ML and heuristics to select the optimal intervention channel
for each patient based on their profile and historical engagement.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np

from ..models.schemas import InterventionChannel, RiskLevel


class TimeOfDay(str, Enum):
    """Time of day categories."""

    MORNING = "morning"  # 6am - 12pm
    AFTERNOON = "afternoon"  # 12pm - 5pm
    EVENING = "evening"  # 5pm - 9pm
    NIGHT = "night"  # 9pm - 6am


@dataclass
class ChannelScore:
    """Score for a channel option."""

    channel: InterventionChannel
    score: float
    confidence: float
    reason: str


@dataclass
class ChannelRecommendation:
    """Recommended channel with timing."""

    primary_channel: InterventionChannel
    backup_channel: Optional[InterventionChannel]
    optimal_time: Optional[time]
    optimal_day_of_week: Optional[int]  # 0=Monday, 6=Sunday
    scores: list[ChannelScore]


class ChannelSelector:
    """
    Selects optimal intervention channel based on:
    - Patient preferences (explicit)
    - Historical engagement data
    - Risk level urgency
    - Time-based patterns
    - Channel availability
    """

    # Base weights for different factors
    FACTOR_WEIGHTS = {
        "patient_preference": 0.30,
        "historical_response": 0.25,
        "channel_effectiveness": 0.20,
        "risk_urgency": 0.15,
        "time_appropriateness": 0.10,
    }

    # Channel effectiveness baselines (from industry data)
    CHANNEL_BASELINES = {
        InterventionChannel.SMS: 0.75,
        InterventionChannel.PUSH_NOTIFICATION: 0.70,
        InterventionChannel.EMAIL: 0.55,
        InterventionChannel.VOICE: 0.45,
        InterventionChannel.CHATBOT: 0.60,
        InterventionChannel.CARE_MANAGER: 0.85,
    }

    # Risk level to channel preferences
    RISK_CHANNEL_PREFERENCES = {
        RiskLevel.LOW: [
            InterventionChannel.SMS,
            InterventionChannel.EMAIL,
            InterventionChannel.PUSH_NOTIFICATION,
        ],
        RiskLevel.MEDIUM: [
            InterventionChannel.SMS,
            InterventionChannel.VOICE,
            InterventionChannel.CHATBOT,
        ],
        RiskLevel.HIGH: [
            InterventionChannel.CARE_MANAGER,
            InterventionChannel.VOICE,
            InterventionChannel.SMS,
        ],
    }

    def __init__(self):
        self.engagement_history: dict[str, list[dict]] = {}

    def select_channel(
        self,
        patient_id: str,
        patient_data: dict[str, Any],
        risk_level: RiskLevel,
        available_channels: Optional[list[InterventionChannel]] = None,
        current_time: Optional[datetime] = None,
    ) -> ChannelRecommendation:
        """
        Select the optimal intervention channel.

        Args:
            patient_id: Patient identifier
            patient_data: Patient profile and preferences
            risk_level: Current risk level
            available_channels: Channels available for this patient
            current_time: Current time for time-based optimization

        Returns:
            ChannelRecommendation with primary and backup channels
        """
        if current_time is None:
            current_time = datetime.now()

        if available_channels is None:
            available_channels = list(InterventionChannel)

        # Filter to only available channels the patient can receive
        patient_channels = self._get_patient_available_channels(
            patient_data, available_channels
        )

        if not patient_channels:
            # Fallback to care manager if nothing else available
            return ChannelRecommendation(
                primary_channel=InterventionChannel.CARE_MANAGER,
                backup_channel=None,
                optimal_time=None,
                optimal_day_of_week=None,
                scores=[
                    ChannelScore(
                        channel=InterventionChannel.CARE_MANAGER,
                        score=1.0,
                        confidence=0.5,
                        reason="No other channels available",
                    )
                ],
            )

        # Calculate scores for each channel
        channel_scores = []
        for channel in patient_channels:
            score = self._calculate_channel_score(
                channel=channel,
                patient_id=patient_id,
                patient_data=patient_data,
                risk_level=risk_level,
                current_time=current_time,
            )
            channel_scores.append(score)

        # Sort by score
        channel_scores.sort(key=lambda x: x.score, reverse=True)

        # Get optimal timing
        optimal_time, optimal_day = self._get_optimal_timing(
            patient_data, channel_scores[0].channel, current_time
        )

        return ChannelRecommendation(
            primary_channel=channel_scores[0].channel,
            backup_channel=channel_scores[1].channel if len(channel_scores) > 1 else None,
            optimal_time=optimal_time,
            optimal_day_of_week=optimal_day,
            scores=channel_scores,
        )

    def _get_patient_available_channels(
        self,
        patient_data: dict[str, Any],
        available_channels: list[InterventionChannel],
    ) -> list[InterventionChannel]:
        """Filter channels based on patient contact info."""
        patient_channels = []

        for channel in available_channels:
            if channel == InterventionChannel.SMS:
                if patient_data.get("phone_number"):
                    patient_channels.append(channel)
            elif channel == InterventionChannel.VOICE:
                if patient_data.get("phone_number"):
                    patient_channels.append(channel)
            elif channel == InterventionChannel.EMAIL:
                if patient_data.get("email"):
                    patient_channels.append(channel)
            elif channel == InterventionChannel.PUSH_NOTIFICATION:
                if patient_data.get("has_mobile_app", False):
                    patient_channels.append(channel)
            elif channel == InterventionChannel.CHATBOT:
                if patient_data.get("has_mobile_app", False) or patient_data.get("email"):
                    patient_channels.append(channel)
            elif channel == InterventionChannel.CARE_MANAGER:
                # Always available as escalation
                patient_channels.append(channel)

        return patient_channels

    def _calculate_channel_score(
        self,
        channel: InterventionChannel,
        patient_id: str,
        patient_data: dict[str, Any],
        risk_level: RiskLevel,
        current_time: datetime,
    ) -> ChannelScore:
        """Calculate score for a specific channel."""
        scores = {}
        reasons = []

        # 1. Patient preference score
        preferred = patient_data.get("preferred_channel")
        if preferred and preferred == channel.value:
            scores["preference"] = 1.0
            reasons.append("Patient's preferred channel")
        elif preferred:
            scores["preference"] = 0.3
        else:
            scores["preference"] = 0.5

        # 2. Historical response score
        history = self.engagement_history.get(patient_id, [])
        channel_history = [h for h in history if h.get("channel") == channel.value]

        if channel_history:
            response_rate = sum(1 for h in channel_history if h.get("responded", False))
            response_rate /= len(channel_history)
            scores["historical"] = response_rate
            if response_rate > 0.7:
                reasons.append("High historical response rate")
        else:
            # No history, use baseline
            scores["historical"] = 0.5

        # 3. Channel effectiveness baseline
        scores["effectiveness"] = self.CHANNEL_BASELINES.get(channel, 0.5)

        # 4. Risk urgency score
        risk_preferences = self.RISK_CHANNEL_PREFERENCES.get(risk_level, [])
        if channel in risk_preferences:
            position = risk_preferences.index(channel)
            scores["risk_urgency"] = 1.0 - (position * 0.2)
            if position == 0:
                reasons.append(f"Top choice for {risk_level.value} risk")
        else:
            scores["risk_urgency"] = 0.3

        # 5. Time appropriateness
        time_score = self._get_time_appropriateness(channel, current_time)
        scores["time"] = time_score
        if time_score > 0.8:
            reasons.append("Good timing for this channel")

        # Calculate weighted score
        final_score = sum(
            scores[key.split("_")[0]] * weight
            for key, weight in self.FACTOR_WEIGHTS.items()
            if key.split("_")[0] in scores
        )

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(patient_data, channel_history)

        return ChannelScore(
            channel=channel,
            score=final_score,
            confidence=confidence,
            reason="; ".join(reasons) if reasons else "Standard selection",
        )

    def _get_time_appropriateness(
        self, channel: InterventionChannel, current_time: datetime
    ) -> float:
        """Score how appropriate the current time is for a channel."""
        hour = current_time.hour

        # Define appropriate hours for each channel
        channel_hours = {
            InterventionChannel.SMS: (8, 21),  # 8am - 9pm
            InterventionChannel.VOICE: (9, 20),  # 9am - 8pm
            InterventionChannel.EMAIL: (0, 24),  # Anytime
            InterventionChannel.PUSH_NOTIFICATION: (7, 22),  # 7am - 10pm
            InterventionChannel.CHATBOT: (0, 24),  # Anytime
            InterventionChannel.CARE_MANAGER: (8, 18),  # Business hours
        }

        start, end = channel_hours.get(channel, (8, 20))

        if start <= hour < end:
            # Within appropriate window
            # Peak scores for mid-window
            mid = (start + end) / 2
            distance_from_mid = abs(hour - mid) / ((end - start) / 2)
            return 1.0 - (distance_from_mid * 0.3)
        else:
            return 0.3  # Outside window but not zero

    def _get_optimal_timing(
        self,
        patient_data: dict[str, Any],
        channel: InterventionChannel,
        current_time: datetime,
    ) -> tuple[Optional[time], Optional[int]]:
        """Determine optimal timing for intervention."""
        # Check patient preference
        preferred_time = patient_data.get("preferred_contact_time")

        if preferred_time:
            time_mapping = {
                "morning": time(9, 0),
                "afternoon": time(14, 0),
                "evening": time(18, 0),
            }
            optimal_time = time_mapping.get(preferred_time, time(10, 0))
        else:
            # Default optimal times by channel
            channel_optimal_times = {
                InterventionChannel.SMS: time(10, 0),
                InterventionChannel.VOICE: time(11, 0),
                InterventionChannel.EMAIL: time(9, 0),
                InterventionChannel.PUSH_NOTIFICATION: time(12, 0),
                InterventionChannel.CHATBOT: time(10, 0),
                InterventionChannel.CARE_MANAGER: time(10, 0),
            }
            optimal_time = channel_optimal_times.get(channel, time(10, 0))

        # Optimal day (avoid weekends for voice, prefer weekdays)
        if channel in (InterventionChannel.VOICE, InterventionChannel.CARE_MANAGER):
            # Prefer Tuesday-Thursday
            optimal_day = 2  # Wednesday
        else:
            optimal_day = None  # Any day

        return optimal_time, optimal_day

    def _calculate_confidence(
        self, patient_data: dict[str, Any], history: list[dict]
    ) -> float:
        """Calculate confidence in channel selection."""
        confidence = 0.5  # Base confidence

        # More history = more confidence
        if len(history) >= 5:
            confidence += 0.2
        elif len(history) >= 2:
            confidence += 0.1

        # Having preference increases confidence
        if patient_data.get("preferred_channel"):
            confidence += 0.15

        # Having contact info increases confidence
        contact_fields = ["phone_number", "email", "has_mobile_app"]
        contact_count = sum(1 for f in contact_fields if patient_data.get(f))
        confidence += contact_count * 0.05

        return min(confidence, 1.0)

    def record_engagement(
        self,
        patient_id: str,
        channel: InterventionChannel,
        responded: bool,
        response_time_hours: Optional[float] = None,
    ) -> None:
        """Record patient engagement for future optimization."""
        if patient_id not in self.engagement_history:
            self.engagement_history[patient_id] = []

        self.engagement_history[patient_id].append({
            "channel": channel.value,
            "responded": responded,
            "response_time_hours": response_time_hours,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Keep only last 20 interactions
        self.engagement_history[patient_id] = self.engagement_history[patient_id][-20:]

    def get_channel_analytics(self) -> dict[str, Any]:
        """Get analytics on channel performance."""
        channel_stats = {}

        for channel in InterventionChannel:
            interactions = []
            for history in self.engagement_history.values():
                interactions.extend([h for h in history if h["channel"] == channel.value])

            if interactions:
                response_rate = sum(1 for i in interactions if i["responded"]) / len(interactions)
                avg_response_time = np.mean([
                    i["response_time_hours"]
                    for i in interactions
                    if i.get("response_time_hours")
                ]) if any(i.get("response_time_hours") for i in interactions) else None

                channel_stats[channel.value] = {
                    "total_interactions": len(interactions),
                    "response_rate": round(response_rate, 3),
                    "avg_response_time_hours": round(avg_response_time, 1) if avg_response_time else None,
                }

        return channel_stats
