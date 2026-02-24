"""
Message template engine for personalized interventions.

Generates contextually appropriate messages based on:
- Patient profile
- Risk factors identified
- Intervention channel
- Time of day
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..models.schemas import BarrierType, InterventionChannel, RiskLevel


class MessageTone(str, Enum):
    """Tone of the message."""

    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    URGENT = "urgent"
    SUPPORTIVE = "supportive"


class MessagePurpose(str, Enum):
    """Purpose of the intervention message."""

    REFILL_REMINDER = "refill_reminder"
    EDUCATION = "education"
    BARRIER_RESOLUTION = "barrier_resolution"
    CHECK_IN = "check_in"
    COST_ASSISTANCE = "cost_assistance"
    SIDE_EFFECT_SUPPORT = "side_effect_support"


@dataclass
class MessageTemplate:
    """A message template with placeholders."""

    template_id: str
    channel: InterventionChannel
    purpose: MessagePurpose
    tone: MessageTone
    subject: Optional[str]  # For email
    body: str
    cta: Optional[str]  # Call to action
    requires_response: bool = False


@dataclass
class PersonalizedMessage:
    """A personalized message ready to send."""

    template_id: str
    channel: InterventionChannel
    subject: Optional[str]
    body: str
    cta: Optional[str]
    metadata: dict[str, Any]


class MessageTemplateEngine:
    """
    Engine for generating personalized intervention messages.

    Supports multiple channels and dynamically selects the best
    template based on context.
    """

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, MessageTemplate]:
        """Load all message templates."""
        templates = {}

        # ============== SMS Templates ==============

        templates["sms_refill_reminder_friendly"] = MessageTemplate(
            template_id="sms_refill_reminder_friendly",
            channel=InterventionChannel.SMS,
            purpose=MessagePurpose.REFILL_REMINDER,
            tone=MessageTone.FRIENDLY,
            subject=None,
            body=(
                "Hi {first_name}! 👋 Time to refill your {medication_name}. "
                "Ready to order? Reply YES or call us at {pharmacy_phone}."
            ),
            cta="Reply YES to refill",
            requires_response=True,
        )

        templates["sms_refill_reminder_urgent"] = MessageTemplate(
            template_id="sms_refill_reminder_urgent",
            channel=InterventionChannel.SMS,
            purpose=MessagePurpose.REFILL_REMINDER,
            tone=MessageTone.URGENT,
            subject=None,
            body=(
                "{first_name}, your {medication_name} ran out {days_overdue} days ago. "
                "Don't miss a dose - reply YES to refill now or call {pharmacy_phone}."
            ),
            cta="Reply YES",
            requires_response=True,
        )

        templates["sms_cost_assistance"] = MessageTemplate(
            template_id="sms_cost_assistance",
            channel=InterventionChannel.SMS,
            purpose=MessagePurpose.COST_ASSISTANCE,
            tone=MessageTone.SUPPORTIVE,
            subject=None,
            body=(
                "Hi {first_name}, we noticed your {medication_name} costs may be high. "
                "Good news: you may qualify for savings! Reply HELP to learn more."
            ),
            cta="Reply HELP",
            requires_response=True,
        )

        templates["sms_side_effect_check"] = MessageTemplate(
            template_id="sms_side_effect_check",
            channel=InterventionChannel.SMS,
            purpose=MessagePurpose.SIDE_EFFECT_SUPPORT,
            tone=MessageTone.SUPPORTIVE,
            subject=None,
            body=(
                "Hi {first_name}, how are you feeling with {medication_name}? "
                "If you have questions about side effects, reply TALK to connect "
                "with a pharmacist."
            ),
            cta="Reply TALK",
            requires_response=True,
        )

        # ============== Email Templates ==============

        templates["email_refill_reminder"] = MessageTemplate(
            template_id="email_refill_reminder",
            channel=InterventionChannel.EMAIL,
            purpose=MessagePurpose.REFILL_REMINDER,
            tone=MessageTone.PROFESSIONAL,
            subject="Time to refill your {medication_name}",
            body="""
Dear {first_name},

This is a friendly reminder that it's time to refill your {medication_name} prescription.

**Prescription Details:**
- Medication: {medication_name}
- Days Supply: {days_supply} days
- Your copay: ${copay_amount}

Staying on track with your medication is important for managing your {condition}.

[Refill Now]({refill_link})

If you have any questions or concerns, please don't hesitate to reach out.

Best regards,
Your OptumRx Care Team

---
Reply to this email or call {pharmacy_phone} for assistance.
            """.strip(),
            cta="Refill Now",
            requires_response=False,
        )

        templates["email_education_diabetes"] = MessageTemplate(
            template_id="email_education_diabetes",
            channel=InterventionChannel.EMAIL,
            purpose=MessagePurpose.EDUCATION,
            tone=MessageTone.SUPPORTIVE,
            subject="Tips for managing your diabetes medications",
            body="""
Dear {first_name},

Managing diabetes is a journey, and we're here to support you every step of the way.

**Why Your Medication Matters**

Your {medication_name} helps control your blood sugar levels, which can:
- Reduce risk of heart disease
- Protect your kidneys and eyes
- Improve your energy levels

**Tips for Success:**

1. **Take it consistently** - Same time each day works best
2. **Don't skip doses** - Even if you feel fine
3. **Monitor your levels** - Track how you're doing

**Your Current Status:**
- Medication coverage: {pdc_percentage}%
- Days until next refill: {days_until_refill}

[Schedule Your Refill]({refill_link})

Questions? Our pharmacists are here to help: {pharmacy_phone}

Take care,
Your OptumRx Care Team
            """.strip(),
            cta="Schedule Your Refill",
            requires_response=False,
        )

        templates["email_cost_assistance_detailed"] = MessageTemplate(
            template_id="email_cost_assistance_detailed",
            channel=InterventionChannel.EMAIL,
            purpose=MessagePurpose.COST_ASSISTANCE,
            tone=MessageTone.SUPPORTIVE,
            subject="Ways to save on your {medication_name}",
            body="""
Dear {first_name},

We understand that medication costs can be a concern. Good news - there may be ways to save on your {medication_name}!

**Potential Savings Options:**

1. **Manufacturer Copay Card**
   You may qualify for a copay assistance program that could reduce your out-of-pocket cost.

2. **Generic Alternative**
   {generic_info}

3. **90-Day Supply**
   Switching to a 90-day mail order supply could save you money on copays.

4. **Patient Assistance Programs**
   Based on your profile, you may qualify for additional assistance.

**Your Current Costs:**
- Current copay: ${copay_amount}
- Potential savings: Up to ${potential_savings}

Want to explore your options?

[Check My Savings Options]({savings_link})

Or call our dedicated team: {pharmacy_phone}

We're here to help you stay healthy without financial stress.

Best regards,
Your OptumRx Care Team
            """.strip(),
            cta="Check My Savings Options",
            requires_response=False,
        )

        # ============== Voice/IVR Templates ==============

        templates["voice_refill_reminder"] = MessageTemplate(
            template_id="voice_refill_reminder",
            channel=InterventionChannel.VOICE,
            purpose=MessagePurpose.REFILL_REMINDER,
            tone=MessageTone.FRIENDLY,
            subject=None,
            body=(
                "Hello {first_name}, this is a courtesy call from OptumRx about your "
                "{medication_name} prescription. It's time for a refill. "
                "Press 1 to refill now. Press 2 to speak with a pharmacist. "
                "Press 3 to hear this message again."
            ),
            cta="Press 1 to refill",
            requires_response=True,
        )

        templates["voice_check_in"] = MessageTemplate(
            template_id="voice_check_in",
            channel=InterventionChannel.VOICE,
            purpose=MessagePurpose.CHECK_IN,
            tone=MessageTone.SUPPORTIVE,
            subject=None,
            body=(
                "Hello {first_name}, this is OptumRx calling to check in on how "
                "you're doing with your {medication_name}. "
                "If you're having any issues with your medication, press 1 to speak "
                "with a pharmacist. Press 2 if everything is going well. "
                "Press 3 to hear this message again."
            ),
            cta="Press 1 for pharmacist",
            requires_response=True,
        )

        # ============== Push Notification Templates ==============

        templates["push_refill_reminder"] = MessageTemplate(
            template_id="push_refill_reminder",
            channel=InterventionChannel.PUSH_NOTIFICATION,
            purpose=MessagePurpose.REFILL_REMINDER,
            tone=MessageTone.FRIENDLY,
            subject="Refill Reminder",
            body="Time to refill {medication_name}! Tap to order now.",
            cta="Order Refill",
            requires_response=False,
        )

        templates["push_refill_ready"] = MessageTemplate(
            template_id="push_refill_ready",
            channel=InterventionChannel.PUSH_NOTIFICATION,
            purpose=MessagePurpose.REFILL_REMINDER,
            tone=MessageTone.FRIENDLY,
            subject="Your Prescription is Ready!",
            body="Your {medication_name} is ready for pickup at {pharmacy_name}.",
            cta="View Details",
            requires_response=False,
        )

        # ============== Care Manager Alert Templates ==============

        templates["care_manager_high_risk_alert"] = MessageTemplate(
            template_id="care_manager_high_risk_alert",
            channel=InterventionChannel.CARE_MANAGER,
            purpose=MessagePurpose.CHECK_IN,
            tone=MessageTone.PROFESSIONAL,
            subject="High Risk Alert: {patient_name}",
            body="""
**HIGH RISK PATIENT ALERT**

Patient: {patient_name} ({patient_id})
Risk Score: {risk_score}/100 ({risk_level})
Medication: {medication_name}

**Top Risk Factors:**
{risk_factors}

**Recommended Actions:**
{recommended_actions}

**Contact Information:**
- Phone: {patient_phone}
- Preferred Contact Time: {preferred_time}

**Recent History:**
- Last fill: {last_fill_date}
- PDC (90-day): {pdc_percentage}%
- Previous interventions: {intervention_count}

Please review and take appropriate action.
            """.strip(),
            cta="Contact Patient",
            requires_response=True,
        )

        return templates

    def get_template(
        self,
        channel: InterventionChannel,
        purpose: MessagePurpose,
        tone: Optional[MessageTone] = None,
    ) -> Optional[MessageTemplate]:
        """
        Find the best matching template.

        Args:
            channel: Target channel
            purpose: Message purpose
            tone: Optional preferred tone

        Returns:
            Best matching template or None
        """
        candidates = [
            t
            for t in self.templates.values()
            if t.channel == channel and t.purpose == purpose
        ]

        if not candidates:
            return None

        if tone:
            tone_match = [t for t in candidates if t.tone == tone]
            if tone_match:
                return tone_match[0]

        return candidates[0]

    def generate_message(
        self,
        template_id: str,
        context: dict[str, Any],
    ) -> PersonalizedMessage:
        """
        Generate a personalized message from a template.

        Args:
            template_id: Template to use
            context: Dictionary with values to interpolate

        Returns:
            PersonalizedMessage ready to send
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Fill in placeholders
        body = self._interpolate(template.body, context)
        subject = self._interpolate(template.subject, context) if template.subject else None
        cta = self._interpolate(template.cta, context) if template.cta else None

        return PersonalizedMessage(
            template_id=template_id,
            channel=template.channel,
            subject=subject,
            body=body,
            cta=cta,
            metadata={
                "purpose": template.purpose.value,
                "tone": template.tone.value,
                "requires_response": template.requires_response,
                "generated_at": datetime.utcnow().isoformat(),
            },
        )

    def generate_adaptive_message(
        self,
        channel: InterventionChannel,
        patient_context: dict[str, Any],
        risk_level: RiskLevel,
        barrier: Optional[BarrierType] = None,
    ) -> PersonalizedMessage:
        """
        Generate an adaptive message based on patient context.

        Automatically selects the best template and tone based on:
        - Risk level
        - Identified barriers
        - Time of day
        - Patient preferences

        Args:
            channel: Target channel
            patient_context: Patient data and history
            risk_level: Current risk level
            barrier: Optional identified barrier

        Returns:
            PersonalizedMessage
        """
        # Determine purpose based on barrier
        if barrier == BarrierType.COST:
            purpose = MessagePurpose.COST_ASSISTANCE
        elif barrier == BarrierType.SIDE_EFFECTS:
            purpose = MessagePurpose.SIDE_EFFECT_SUPPORT
        elif barrier in (BarrierType.LACK_OF_UNDERSTANDING, BarrierType.BELIEFS):
            purpose = MessagePurpose.EDUCATION
        else:
            purpose = MessagePurpose.REFILL_REMINDER

        # Determine tone based on risk level
        if risk_level == RiskLevel.HIGH:
            tone = MessageTone.URGENT
        elif risk_level == RiskLevel.MEDIUM:
            tone = MessageTone.FRIENDLY
        else:
            tone = MessageTone.FRIENDLY

        # Find template
        template = self.get_template(channel, purpose, tone)
        if not template:
            # Fallback to refill reminder
            template = self.get_template(channel, MessagePurpose.REFILL_REMINDER)

        if not template:
            raise ValueError(f"No template available for channel: {channel}")

        return self.generate_message(template.template_id, patient_context)

    def _interpolate(self, text: str, context: dict[str, Any]) -> str:
        """Interpolate placeholders in text with context values."""
        if not text:
            return text

        result = text
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        return result

    def list_templates(
        self, channel: Optional[InterventionChannel] = None
    ) -> list[dict[str, Any]]:
        """List available templates, optionally filtered by channel."""
        templates = []
        for template in self.templates.values():
            if channel and template.channel != channel:
                continue
            templates.append({
                "template_id": template.template_id,
                "channel": template.channel.value,
                "purpose": template.purpose.value,
                "tone": template.tone.value,
                "requires_response": template.requires_response,
            })
        return templates
