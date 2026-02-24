"""
Conversational AI Agent for Barrier Resolution.

Uses Claude (Anthropic) to have intelligent conversations with patients,
identify adherence barriers, and provide personalized support.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from ..models.schemas import (
    BarrierType,
    ChatRequest,
    ChatResponse,
    ConversationContext,
    ConversationMessage,
)

logger = structlog.get_logger()


@dataclass
class MedicationInfo:
    """Information about a medication for RAG context."""

    ndc: str
    name: str
    generic_name: Optional[str]
    drug_class: str
    common_uses: list[str]
    common_side_effects: list[str]
    administration_tips: list[str]
    food_interactions: Optional[str]
    copay_assistance_available: bool


@dataclass
class AssistanceProgram:
    """Copay assistance program information."""

    program_id: str
    name: str
    medication_names: list[str]
    eligibility_criteria: str
    potential_savings: str
    enrollment_url: Optional[str]
    phone_number: Optional[str]


class ConversationalAgent:
    """
    AI-powered conversational agent for medication adherence support.

    Capabilities:
    - Identifies adherence barriers through natural conversation
    - Provides medication education
    - Finds copay assistance programs
    - Answers common questions
    - Escalates complex issues to pharmacists/care managers
    """

    SYSTEM_PROMPT = """You are a friendly and empathetic medication support assistant for OptumRx.
Your goal is to help patients stay on track with their medications by:

1. IDENTIFYING BARRIERS: Listen carefully to understand why patients might be struggling with their medications.
   Common barriers include: cost concerns, side effects, forgetfulness, complexity, lack of understanding.

2. PROVIDING SUPPORT: Offer helpful information and solutions based on the barrier identified:
   - Cost: Mention copay assistance programs, generic alternatives, 90-day supplies
   - Side effects: Provide tips, suggest talking to pharmacist/doctor
   - Forgetfulness: Offer to set up reminders, suggest pill organizers
   - Complexity: Explain the regimen clearly, offer simplification tips
   - Understanding: Educate about why the medication is important

3. BEING EMPATHETIC: Use warm, supportive language. Acknowledge their feelings.
   Never be judgmental about non-adherence.

4. KNOWING YOUR LIMITS: If a question is clinical/medical, recommend speaking with a pharmacist or doctor.
   Don't provide specific medical advice.

5. TAKING ACTION: When appropriate, offer to help schedule refills, set reminders, or connect them with resources.

Keep responses concise (2-4 sentences typically) unless more detail is needed.
Always end with an open question or clear next step when appropriate.

Current patient context will be provided in the conversation."""

    BARRIER_DETECTION_PROMPT = """Based on this patient message, identify if they're expressing any medication adherence barriers.

Barriers to look for:
- COST: Mentions price, expense, afford, copay, insurance, money
- FORGETFULNESS: Mentions forgetting, remember, skip, miss doses
- SIDE_EFFECTS: Mentions feeling bad, nausea, dizzy, symptoms from medication
- COMPLEXITY: Mentions confusing, too many pills, complicated schedule
- LACK_OF_UNDERSTANDING: Doesn't understand why they need it, questions if it's working
- ACCESS: Can't get to pharmacy, transportation issues
- BELIEFS: Doesn't believe medication helps, prefers natural remedies

Respond with JSON only: {"barrier": "BARRIER_TYPE" or null, "confidence": 0.0-1.0, "reasoning": "brief explanation"}

Patient message: "{message}"
"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        self.api_key = anthropic_api_key
        self.model = model
        self._client = None

        # In-memory conversation storage (use Redis/DB in production)
        self.conversations: dict[str, ConversationContext] = {}

        # Knowledge base (simplified - would use vector DB in production)
        self.medication_kb = self._load_medication_kb()
        self.assistance_programs = self._load_assistance_programs()

        self._log = logger.bind(service="conversational_agent")

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                self._log.warning("anthropic_not_installed")
                return None
            except Exception as e:
                self._log.error("anthropic_init_error", error=str(e))
                return None
        return self._client

    def _load_medication_kb(self) -> dict[str, MedicationInfo]:
        """Load medication knowledge base."""
        # Simplified KB - in production, this would be a vector DB
        return {
            "metformin": MedicationInfo(
                ndc="00093-7212-01",
                name="Metformin",
                generic_name=None,
                drug_class="Biguanide",
                common_uses=["Type 2 Diabetes"],
                common_side_effects=["Nausea", "Stomach upset", "Diarrhea"],
                administration_tips=[
                    "Take with food to reduce stomach upset",
                    "Stay hydrated",
                    "Take at the same time each day",
                ],
                food_interactions="Avoid excessive alcohol",
                copay_assistance_available=True,
            ),
            "lisinopril": MedicationInfo(
                ndc="00093-7180-01",
                name="Lisinopril",
                generic_name=None,
                drug_class="ACE Inhibitor",
                common_uses=["High Blood Pressure", "Heart Failure"],
                common_side_effects=["Dry cough", "Dizziness", "Headache"],
                administration_tips=[
                    "Take at the same time each day",
                    "Rise slowly to avoid dizziness",
                    "Avoid potassium supplements without doctor approval",
                ],
                food_interactions="Limit potassium-rich foods",
                copay_assistance_available=True,
            ),
            "atorvastatin": MedicationInfo(
                ndc="00093-5057-01",
                name="Atorvastatin",
                generic_name="Lipitor",
                drug_class="Statin",
                common_uses=["High Cholesterol", "Heart Disease Prevention"],
                common_side_effects=["Muscle pain", "Joint pain", "Digestive issues"],
                administration_tips=[
                    "Can be taken with or without food",
                    "Report any unexplained muscle pain",
                    "Avoid grapefruit juice",
                ],
                food_interactions="Avoid grapefruit and grapefruit juice",
                copay_assistance_available=True,
            ),
        }

    def _load_assistance_programs(self) -> list[AssistanceProgram]:
        """Load copay assistance programs."""
        return [
            AssistanceProgram(
                program_id="rx_assist_diabetes",
                name="Diabetes Care Savings Program",
                medication_names=["Metformin", "Insulin", "Januvia"],
                eligibility_criteria="Commercial insurance, income requirements may apply",
                potential_savings="Up to $150/month",
                enrollment_url="https://example.com/diabetes-savings",
                phone_number="1-800-555-0100",
            ),
            AssistanceProgram(
                program_id="heart_health_assist",
                name="Heart Health Assistance",
                medication_names=["Lisinopril", "Atorvastatin", "Amlodipine"],
                eligibility_criteria="Most patients with commercial insurance qualify",
                potential_savings="Up to $100/month",
                enrollment_url="https://example.com/heart-savings",
                phone_number="1-800-555-0101",
            ),
        ]

    async def chat(
        self,
        request: ChatRequest,
        patient_context: Optional[dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        Process a chat message and generate response.

        Args:
            request: Chat request with patient message
            patient_context: Optional patient data for personalization

        Returns:
            ChatResponse with AI response and detected barriers
        """
        # Get or create conversation
        if request.conversation_id and request.conversation_id in self.conversations:
            conversation = self.conversations[request.conversation_id]
        else:
            conversation_id = request.conversation_id or str(uuid.uuid4())
            conversation = ConversationContext(
                patient_id=request.patient_id,
                conversation_id=conversation_id,
                messages=[],
            )
            self.conversations[conversation_id] = conversation

        # Add user message
        conversation.messages.append(
            ConversationMessage(
                role="user",
                content=request.message,
                timestamp=datetime.utcnow(),
            )
        )

        # Detect barriers
        barrier, barrier_confidence = await self._detect_barrier(request.message)
        if barrier and barrier_confidence > 0.6:
            conversation.identified_barriers.append(barrier)

        # Detect sentiment
        sentiment = self._analyze_sentiment(request.message)

        # Generate response
        response_text = await self._generate_response(
            conversation=conversation,
            patient_context=patient_context,
            detected_barrier=barrier,
        )

        # Add assistant response
        conversation.messages.append(
            ConversationMessage(
                role="assistant",
                content=response_text,
                timestamp=datetime.utcnow(),
            )
        )

        # Check if escalation needed
        requires_escalation = self._check_escalation_needed(conversation, sentiment)
        if requires_escalation:
            conversation.requires_escalation = True
            conversation.escalation_type = "pharmacist"

        # Determine suggested action
        suggested_action = self._get_suggested_action(barrier, patient_context)

        return ChatResponse(
            conversation_id=conversation.conversation_id,
            response=response_text,
            identified_barrier=barrier,
            suggested_action=suggested_action,
            requires_human_followup=requires_escalation,
            sentiment=sentiment,
        )

    async def _detect_barrier(
        self, message: str
    ) -> tuple[Optional[BarrierType], float]:
        """Detect adherence barrier from message using LLM."""
        client = self._get_client()

        if client:
            try:
                prompt = self.BARRIER_DETECTION_PROMPT.format(message=message)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )

                result_text = response.content[0].text
                # Parse JSON response
                result = json.loads(result_text)

                barrier_str = result.get("barrier")
                confidence = result.get("confidence", 0.5)

                if barrier_str:
                    barrier_map = {
                        "COST": BarrierType.COST,
                        "FORGETFULNESS": BarrierType.FORGETFULNESS,
                        "SIDE_EFFECTS": BarrierType.SIDE_EFFECTS,
                        "COMPLEXITY": BarrierType.COMPLEXITY,
                        "LACK_OF_UNDERSTANDING": BarrierType.LACK_OF_UNDERSTANDING,
                        "ACCESS": BarrierType.ACCESS,
                        "BELIEFS": BarrierType.BELIEFS,
                    }
                    return barrier_map.get(barrier_str), confidence

            except Exception as e:
                self._log.error("barrier_detection_error", error=str(e))

        # Fallback: Simple keyword detection
        return self._keyword_barrier_detection(message)

    def _keyword_barrier_detection(
        self, message: str
    ) -> tuple[Optional[BarrierType], float]:
        """Simple keyword-based barrier detection as fallback."""
        message_lower = message.lower()

        barrier_keywords = {
            BarrierType.COST: ["expensive", "cost", "afford", "price", "copay", "money", "pay"],
            BarrierType.FORGETFULNESS: ["forget", "remember", "skip", "miss", "forgot"],
            BarrierType.SIDE_EFFECTS: [
                "sick", "nausea", "tired", "dizzy", "headache", "pain",
                "side effect", "feel bad", "stomach",
            ],
            BarrierType.COMPLEXITY: ["confus", "complicated", "too many", "hard to"],
            BarrierType.LACK_OF_UNDERSTANDING: [
                "why do i need", "does it work", "don't understand",
                "what does it do", "not sure why",
            ],
        }

        for barrier, keywords in barrier_keywords.items():
            if any(kw in message_lower for kw in keywords):
                return barrier, 0.7

        return None, 0.0

    def _analyze_sentiment(self, message: str) -> str:
        """Simple sentiment analysis."""
        message_lower = message.lower()

        negative_words = [
            "frustrated", "angry", "upset", "hate", "terrible",
            "worst", "awful", "annoyed", "fed up",
        ]
        positive_words = [
            "thank", "great", "good", "helpful", "appreciate",
            "better", "happy", "glad",
        ]

        neg_count = sum(1 for w in negative_words if w in message_lower)
        pos_count = sum(1 for w in positive_words if w in message_lower)

        if neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        return "neutral"

    async def _generate_response(
        self,
        conversation: ConversationContext,
        patient_context: Optional[dict[str, Any]],
        detected_barrier: Optional[BarrierType],
    ) -> str:
        """Generate AI response using Claude."""
        client = self._get_client()

        # Build context message
        context_parts = []
        if patient_context:
            context_parts.append(f"Patient: {patient_context.get('first_name', 'Unknown')}")
            if patient_context.get("medication_name"):
                context_parts.append(f"Medication: {patient_context['medication_name']}")
            if patient_context.get("days_since_last_fill"):
                context_parts.append(
                    f"Days since last fill: {patient_context['days_since_last_fill']}"
                )

        if detected_barrier:
            context_parts.append(f"Detected barrier: {detected_barrier.value}")

        context_str = "\n".join(context_parts) if context_parts else "No additional context"

        # Build messages for API
        messages = []

        # Add conversation history
        for msg in conversation.messages[-10:]:  # Last 10 messages
            messages.append({
                "role": msg.role if msg.role != "system" else "user",
                "content": msg.content,
            })

        if client:
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=f"{self.SYSTEM_PROMPT}\n\nPatient Context:\n{context_str}",
                    messages=messages,
                )
                return response.content[0].text

            except Exception as e:
                self._log.error("response_generation_error", error=str(e))

        # Fallback response
        return self._generate_fallback_response(detected_barrier, patient_context)

    def _generate_fallback_response(
        self,
        barrier: Optional[BarrierType],
        patient_context: Optional[dict[str, Any]],
    ) -> str:
        """Generate fallback response when LLM is unavailable."""
        first_name = patient_context.get("first_name", "there") if patient_context else "there"

        if barrier == BarrierType.COST:
            return (
                f"Hi {first_name}, I understand cost can be a concern. "
                "Good news - there may be savings programs available for your medication. "
                "Would you like me to check what options might help reduce your costs?"
            )
        elif barrier == BarrierType.FORGETFULNESS:
            return (
                f"Hi {first_name}, it's common to occasionally forget doses. "
                "Would you like me to help set up reminders? We can also look at "
                "options like a 90-day supply to reduce how often you need to refill."
            )
        elif barrier == BarrierType.SIDE_EFFECTS:
            return (
                f"I'm sorry to hear you're experiencing side effects, {first_name}. "
                "Some side effects improve over time, but it's important to discuss this "
                "with a pharmacist or doctor. Would you like me to connect you with a pharmacist?"
            )
        elif barrier == BarrierType.COMPLEXITY:
            return (
                f"Managing multiple medications can be challenging, {first_name}. "
                "A pill organizer can really help. Would you like some tips on organizing "
                "your medication schedule?"
            )
        elif barrier == BarrierType.LACK_OF_UNDERSTANDING:
            medication = patient_context.get("medication_name", "your medication") if patient_context else "your medication"
            return (
                f"Great question! Understanding why you take {medication} is important. "
                "Would you like me to explain how it helps your health?"
            )
        else:
            return (
                f"Hi {first_name}! I'm here to help with any questions about your medications. "
                "Is there anything specific you'd like to know or any concerns I can help address?"
            )

    def _check_escalation_needed(
        self,
        conversation: ConversationContext,
        sentiment: str,
    ) -> bool:
        """Check if conversation should be escalated to human."""
        # Escalate if:
        # 1. Multiple barriers identified
        if len(set(conversation.identified_barriers)) >= 2:
            return True

        # 2. Negative sentiment detected multiple times
        negative_count = sum(
            1 for msg in conversation.messages
            if msg.role == "user" and self._analyze_sentiment(msg.content) == "negative"
        )
        if negative_count >= 2:
            return True

        # 3. Side effects barrier (clinical concern)
        if BarrierType.SIDE_EFFECTS in conversation.identified_barriers:
            return True

        # 4. Explicit request for human
        last_message = conversation.messages[-1].content.lower() if conversation.messages else ""
        human_requests = ["speak to someone", "talk to a person", "pharmacist", "doctor", "human"]
        if any(req in last_message for req in human_requests):
            return True

        return False

    def _get_suggested_action(
        self,
        barrier: Optional[BarrierType],
        patient_context: Optional[dict[str, Any]],
    ) -> Optional[str]:
        """Get suggested action based on barrier."""
        if not barrier:
            return None

        actions = {
            BarrierType.COST: "check_copay_assistance",
            BarrierType.FORGETFULNESS: "setup_reminders",
            BarrierType.SIDE_EFFECTS: "connect_pharmacist",
            BarrierType.COMPLEXITY: "simplify_regimen",
            BarrierType.LACK_OF_UNDERSTANDING: "provide_education",
            BarrierType.ACCESS: "arrange_delivery",
        }

        return actions.get(barrier)

    def find_assistance_programs(
        self, medication_name: str
    ) -> list[AssistanceProgram]:
        """Find copay assistance programs for a medication."""
        medication_lower = medication_name.lower()
        return [
            program
            for program in self.assistance_programs
            if any(med.lower() in medication_lower or medication_lower in med.lower()
                   for med in program.medication_names)
        ]

    def get_medication_info(self, medication_name: str) -> Optional[MedicationInfo]:
        """Get information about a medication."""
        return self.medication_kb.get(medication_name.lower())

    def get_conversation_summary(
        self, conversation_id: str
    ) -> Optional[dict[str, Any]]:
        """Get summary of a conversation."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None

        return {
            "conversation_id": conversation_id,
            "patient_id": conversation.patient_id,
            "message_count": len(conversation.messages),
            "barriers_identified": [b.value for b in conversation.identified_barriers],
            "requires_escalation": conversation.requires_escalation,
            "escalation_type": conversation.escalation_type,
            "suggested_actions": conversation.suggested_actions,
        }
