"""
Generate synthetic data for demonstration and testing.

Creates realistic but fake patient, medication, and intervention data
that mimics real-world patterns seen in medication adherence programs.
"""
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


# ============== Configuration ==============

NUM_PATIENTS = 500
NUM_FILLS_PER_PATIENT_RANGE = (6, 24)  # Range of fills per patient
NUM_INTERVENTIONS = 1000
OUTPUT_DIR = Path(__file__).parent


# ============== Reference Data ==============

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]

MEDICATIONS = [
    {"name": "Metformin 500mg", "ndc": "00093-7212-01", "class": "Diabetes", "days_supply": [30, 90], "copay_range": (5, 25)},
    {"name": "Metformin 1000mg", "ndc": "00093-7214-01", "class": "Diabetes", "days_supply": [30, 90], "copay_range": (5, 30)},
    {"name": "Lisinopril 10mg", "ndc": "00093-7180-01", "class": "Hypertension", "days_supply": [30, 90], "copay_range": (3, 15)},
    {"name": "Lisinopril 20mg", "ndc": "00093-7181-01", "class": "Hypertension", "days_supply": [30, 90], "copay_range": (3, 20)},
    {"name": "Atorvastatin 20mg", "ndc": "00093-5057-01", "class": "Cholesterol", "days_supply": [30, 90], "copay_range": (5, 25)},
    {"name": "Atorvastatin 40mg", "ndc": "00093-5058-01", "class": "Cholesterol", "days_supply": [30, 90], "copay_range": (8, 35)},
    {"name": "Amlodipine 5mg", "ndc": "00093-3161-01", "class": "Hypertension", "days_supply": [30, 90], "copay_range": (3, 15)},
    {"name": "Amlodipine 10mg", "ndc": "00093-3162-01", "class": "Hypertension", "days_supply": [30, 90], "copay_range": (5, 20)},
    {"name": "Omeprazole 20mg", "ndc": "00093-5291-01", "class": "GERD", "days_supply": [30], "copay_range": (10, 45)},
    {"name": "Losartan 50mg", "ndc": "00093-7365-01", "class": "Hypertension", "days_supply": [30, 90], "copay_range": (5, 20)},
    {"name": "Gabapentin 300mg", "ndc": "00093-0637-01", "class": "Pain/Seizures", "days_supply": [30], "copay_range": (5, 25)},
    {"name": "Sertraline 50mg", "ndc": "00093-4561-01", "class": "Mental Health", "days_supply": [30], "copay_range": (5, 20)},
]

DIAGNOSIS_CODES = {
    "Diabetes": ["E11.9", "E11.65", "E11.21"],
    "Hypertension": ["I10", "I11.9", "I12.9"],
    "Cholesterol": ["E78.0", "E78.1", "E78.5"],
    "GERD": ["K21.0", "K21.9"],
    "Pain/Seizures": ["G40.909", "M54.5"],
    "Mental Health": ["F32.9", "F33.0", "F41.1"],
}

PLAN_TYPES = ["commercial", "medicare", "medicaid", "exchange"]
GENDERS = ["M", "F"]
CHANNELS = ["sms", "email", "voice", "push_notification", "care_manager"]
INTERVENTION_STATUSES = ["sent", "delivered", "responded", "successful", "failed"]


# ============== Generator Functions ==============

def generate_patient(patient_idx: int) -> dict[str, Any]:
    """Generate a synthetic patient."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)

    # Age distribution - more older patients (typical for chronic conditions)
    age = int(np.clip(np.random.normal(58, 15), 25, 90))

    # Plan type depends on age
    if age >= 65:
        plan_type = random.choices(
            PLAN_TYPES, weights=[0.1, 0.7, 0.15, 0.05]
        )[0]
    else:
        plan_type = random.choices(
            PLAN_TYPES, weights=[0.6, 0.05, 0.2, 0.15]
        )[0]

    # Select 1-3 conditions
    num_conditions = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    conditions = random.sample(list(DIAGNOSIS_CODES.keys()), num_conditions)
    diagnosis_codes = []
    for condition in conditions:
        diagnosis_codes.extend(random.sample(DIAGNOSIS_CODES[condition], 1))

    # Contact preferences
    has_phone = random.random() > 0.05
    has_email = random.random() > 0.15
    has_app = random.random() > 0.6

    preferred_channel = None
    if has_phone or has_email or has_app:
        available_channels = []
        if has_phone:
            available_channels.extend(["sms", "voice"])
        if has_email:
            available_channels.append("email")
        if has_app:
            available_channels.append("push_notification")
        preferred_channel = random.choice(available_channels) if available_channels else None

    return {
        "patient_id": f"P{str(patient_idx).zfill(6)}",
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "gender": random.choice(GENDERS),
        "plan_type": plan_type,
        "zip_code": f"{random.randint(10000, 99999)}",
        "diagnosis_codes": diagnosis_codes,
        "conditions": conditions,
        "phone_number": f"+1{random.randint(2000000000, 9999999999)}" if has_phone else None,
        "email": f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}@email.com" if has_email else None,
        "has_mobile_app": has_app,
        "preferred_channel": preferred_channel,
        "preferred_contact_time": random.choice(["morning", "afternoon", "evening", None]),
        "created_at": (datetime.now() - timedelta(days=random.randint(30, 730))).isoformat(),
    }


def generate_medication_fills(patient: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate medication fill history for a patient."""
    fills = []

    # Select medications based on conditions
    patient_conditions = patient.get("conditions", ["Diabetes"])
    eligible_meds = [m for m in MEDICATIONS if m["class"] in patient_conditions]
    if not eligible_meds:
        eligible_meds = MEDICATIONS[:2]

    # Select 1-3 medications
    num_meds = min(len(eligible_meds), random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0])
    selected_meds = random.sample(eligible_meds, num_meds)

    # Determine adherence pattern for this patient
    # Some patients are highly adherent, some are not
    adherence_profile = random.choices(
        ["high", "medium", "low"],
        weights=[0.4, 0.4, 0.2]
    )[0]

    adherence_params = {
        "high": {"gap_prob": 0.05, "gap_days_range": (1, 5)},
        "medium": {"gap_prob": 0.2, "gap_days_range": (3, 14)},
        "low": {"gap_prob": 0.4, "gap_days_range": (7, 45)},
    }

    params = adherence_params[adherence_profile]

    for med in selected_meds:
        # Generate fill history going back 12-18 months
        start_date = date.today() - timedelta(days=random.randint(365, 540))
        current_date = start_date
        refill_number = 0

        while current_date <= date.today():
            days_supply = random.choice(med["days_supply"])
            copay = round(random.uniform(*med["copay_range"]), 2)

            fill = {
                "fill_id": f"F{len(fills):08d}",
                "patient_id": patient["patient_id"],
                "medication_ndc": med["ndc"],
                "medication_name": med["name"],
                "fill_date": current_date.isoformat(),
                "days_supply": days_supply,
                "refill_number": refill_number,
                "copay_amount": copay,
                "quantity": days_supply,  # Simplified
                "pharmacy_npi": f"1{random.randint(100000000, 999999999)}",
                "prescriber_npi": f"1{random.randint(100000000, 999999999)}",
            }

            fills.append(fill)
            refill_number += 1

            # Calculate next fill date
            gap_days = 0
            if random.random() < params["gap_prob"]:
                gap_days = random.randint(*params["gap_days_range"])

            current_date = current_date + timedelta(days=days_supply + gap_days)

    return fills


def generate_intervention(
    intervention_idx: int,
    patients: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a synthetic intervention record."""
    patient = random.choice(patients)

    # Weight channels by patient availability
    available_channels = []
    if patient.get("phone_number"):
        available_channels.extend(["sms", "voice"])
    if patient.get("email"):
        available_channels.append("email")
    if patient.get("has_mobile_app"):
        available_channels.append("push_notification")
    available_channels.append("care_manager")  # Always available

    channel = random.choice(available_channels)

    # Status depends on channel
    if channel == "care_manager":
        status = random.choices(
            INTERVENTION_STATUSES,
            weights=[0.05, 0.1, 0.3, 0.5, 0.05]
        )[0]
    elif channel in ["sms", "push_notification"]:
        status = random.choices(
            INTERVENTION_STATUSES,
            weights=[0.1, 0.3, 0.25, 0.3, 0.05]
        )[0]
    else:
        status = random.choices(
            INTERVENTION_STATUSES,
            weights=[0.15, 0.25, 0.2, 0.25, 0.15]
        )[0]

    sent_date = date.today() - timedelta(days=random.randint(0, 90))

    response_time_hours = None
    refill_completed = False
    refill_date = None

    if status in ["responded", "successful"]:
        response_time_hours = round(np.random.exponential(24), 1)

        if status == "successful":
            refill_completed = True
            refill_date = (sent_date + timedelta(days=random.randint(1, 7))).isoformat()

    return {
        "intervention_id": f"I{intervention_idx:08d}",
        "patient_id": patient["patient_id"],
        "channel": channel,
        "status": status,
        "sent_at": datetime.combine(sent_date, datetime.min.time()).isoformat(),
        "response_time_hours": response_time_hours,
        "refill_completed": refill_completed,
        "refill_date": refill_date,
        "message_template_id": f"template_{channel}_001",
        "risk_score_at_intervention": round(random.uniform(30, 90), 1),
    }


def calculate_patient_metrics(
    patient: dict[str, Any],
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate adherence metrics for a patient."""
    patient_fills = [f for f in fills if f["patient_id"] == patient["patient_id"]]

    if not patient_fills:
        return {
            "patient_id": patient["patient_id"],
            "pdc_90_days": 0.0,
            "pdc_180_days": 0.0,
            "total_fills": 0,
            "days_since_last_fill": 999,
            "gap_count": 0,
            "risk_score": 75.0,
        }

    # Sort by date
    patient_fills.sort(key=lambda x: x["fill_date"])

    # Calculate PDC (simplified)
    today = date.today()
    days_90 = today - timedelta(days=90)
    days_180 = today - timedelta(days=180)

    recent_fills = [f for f in patient_fills if date.fromisoformat(f["fill_date"]) >= days_90]
    covered_days_90 = sum(f["days_supply"] for f in recent_fills)
    pdc_90 = min(covered_days_90 / 90, 1.0)

    recent_fills_180 = [f for f in patient_fills if date.fromisoformat(f["fill_date"]) >= days_180]
    covered_days_180 = sum(f["days_supply"] for f in recent_fills_180)
    pdc_180 = min(covered_days_180 / 180, 1.0)

    # Days since last fill
    last_fill_date = date.fromisoformat(patient_fills[-1]["fill_date"])
    days_since_last = (today - last_fill_date).days

    # Count gaps
    gap_count = 0
    for i in range(len(patient_fills) - 1):
        current = patient_fills[i]
        next_fill = patient_fills[i + 1]
        coverage_end = date.fromisoformat(current["fill_date"]) + timedelta(days=current["days_supply"])
        next_date = date.fromisoformat(next_fill["fill_date"])
        if (next_date - coverage_end).days > 7:
            gap_count += 1

    # Calculate risk score (heuristic)
    risk_score = 30.0

    # PDC impact
    if pdc_90 < 0.5:
        risk_score += 35
    elif pdc_90 < 0.8:
        risk_score += 20

    # Gap impact
    risk_score += min(gap_count * 5, 20)

    # Days since last fill
    if days_since_last > 30:
        risk_score += 15
    elif days_since_last > 14:
        risk_score += 8

    # Age factor (elderly slightly higher risk)
    if patient["age"] > 75:
        risk_score += 5

    # Depression diagnosis increases risk
    if any("F32" in d or "F33" in d for d in patient.get("diagnosis_codes", [])):
        risk_score += 10

    risk_score = min(risk_score, 100)

    return {
        "patient_id": patient["patient_id"],
        "pdc_90_days": round(pdc_90, 3),
        "pdc_180_days": round(pdc_180, 3),
        "total_fills": len(patient_fills),
        "days_since_last_fill": days_since_last,
        "gap_count": gap_count,
        "risk_score": round(risk_score, 1),
        "risk_level": "high" if risk_score >= 70 else "medium" if risk_score >= 30 else "low",
    }


def main():
    """Generate all synthetic data."""
    print("Generating synthetic data...")

    # Generate patients
    print(f"  Generating {NUM_PATIENTS} patients...")
    patients = [generate_patient(i) for i in range(NUM_PATIENTS)]

    # Generate medication fills
    print("  Generating medication fills...")
    all_fills = []
    for patient in patients:
        fills = generate_medication_fills(patient)
        all_fills.extend(fills)
    print(f"    Generated {len(all_fills)} fills")

    # Generate interventions
    print(f"  Generating {NUM_INTERVENTIONS} interventions...")
    interventions = [generate_intervention(i, patients) for i in range(NUM_INTERVENTIONS)]

    # Calculate patient metrics
    print("  Calculating patient metrics...")
    patient_metrics = [calculate_patient_metrics(p, all_fills) for p in patients]

    # Merge metrics into patients
    metrics_dict = {m["patient_id"]: m for m in patient_metrics}
    for patient in patients:
        metrics = metrics_dict.get(patient["patient_id"], {})
        patient.update({
            "pdc_90_days": metrics.get("pdc_90_days", 0),
            "pdc_180_days": metrics.get("pdc_180_days", 0),
            "total_fills": metrics.get("total_fills", 0),
            "days_since_last_fill": metrics.get("days_since_last_fill", 999),
            "gap_count": metrics.get("gap_count", 0),
            "risk_score": metrics.get("risk_score", 50),
            "risk_level": metrics.get("risk_level", "medium"),
        })

    # Save to files
    print("  Saving files...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "patients.json", "w") as f:
        json.dump(patients, f, indent=2)
    print(f"    Saved patients.json ({len(patients)} records)")

    with open(OUTPUT_DIR / "medication_fills.json", "w") as f:
        json.dump(all_fills, f, indent=2)
    print(f"    Saved medication_fills.json ({len(all_fills)} records)")

    with open(OUTPUT_DIR / "interventions.json", "w") as f:
        json.dump(interventions, f, indent=2)
    print(f"    Saved interventions.json ({len(interventions)} records)")

    # Generate summary statistics
    summary = {
        "generated_at": datetime.now().isoformat(),
        "patients": {
            "total": len(patients),
            "high_risk": len([p for p in patients if p["risk_level"] == "high"]),
            "medium_risk": len([p for p in patients if p["risk_level"] == "medium"]),
            "low_risk": len([p for p in patients if p["risk_level"] == "low"]),
            "avg_age": round(sum(p["age"] for p in patients) / len(patients), 1),
            "avg_pdc_90": round(sum(p["pdc_90_days"] for p in patients) / len(patients), 3),
        },
        "fills": {
            "total": len(all_fills),
            "avg_per_patient": round(len(all_fills) / len(patients), 1),
        },
        "interventions": {
            "total": len(interventions),
            "by_channel": {
                ch: len([i for i in interventions if i["channel"] == ch])
                for ch in CHANNELS
            },
            "by_status": {
                st: len([i for i in interventions if i["status"] == st])
                for st in INTERVENTION_STATUSES
            },
        },
    }

    with open(OUTPUT_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved data_summary.json")

    print("\nSynthetic data generation complete!")
    print(f"\nSummary:")
    print(f"  Total patients: {summary['patients']['total']}")
    print(f"    High risk: {summary['patients']['high_risk']}")
    print(f"    Medium risk: {summary['patients']['medium_risk']}")
    print(f"    Low risk: {summary['patients']['low_risk']}")
    print(f"  Average PDC (90 days): {summary['patients']['avg_pdc_90']:.1%}")
    print(f"  Total medication fills: {summary['fills']['total']}")
    print(f"  Total interventions: {summary['interventions']['total']}")


if __name__ == "__main__":
    main()
