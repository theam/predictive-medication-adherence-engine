"""
Feature engineering for medication adherence prediction.

Transforms raw patient and medication data into ML-ready features.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    lookback_days: int = 365
    min_fills_required: int = 2
    gap_threshold_days: int = 7
    high_risk_diagnosis_codes: tuple = (
        "F32",  # Depression
        "F33",  # Recurrent depression
        "F41",  # Anxiety disorders
        "E11",  # Type 2 diabetes
        "I10",  # Essential hypertension
    )
    chronic_medication_days_supply_threshold: int = 28


class FeatureEngineer:
    """
    Feature engineering pipeline for adherence prediction.

    Transforms raw data into 50+ features across categories:
    - Refill history features
    - Demographic features
    - Medication complexity features
    - Cost features
    - Clinical risk features
    - Behavioral features
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, patient_data: pd.DataFrame, fill_data: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit the feature engineer on training data.

        Args:
            patient_data: DataFrame with patient demographics
            fill_data: DataFrame with medication fill history

        Returns:
            Self for method chaining
        """
        # Fit label encoders for categorical variables
        categorical_cols = ["gender", "plan_type", "preferred_channel"]
        for col in categorical_cols:
            if col in patient_data.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(patient_data[col].fillna("unknown"))

        self._is_fitted = True
        return self

    def transform(
        self,
        patient_data: pd.DataFrame,
        fill_data: pd.DataFrame,
        reference_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Transform raw data into features.

        Args:
            patient_data: DataFrame with patient demographics
            fill_data: DataFrame with medication fill history
            reference_date: Date to calculate features from (default: today)

        Returns:
            DataFrame with engineered features
        """
        if reference_date is None:
            reference_date = date.today()

        features_list = []

        for _, patient in patient_data.iterrows():
            if fill_data.empty or "patient_id" not in fill_data.columns:
                patient_fills = pd.DataFrame()
            else:
                patient_fills = fill_data[fill_data["patient_id"] == patient["patient_id"]]
            features = self._engineer_patient_features(patient, patient_fills, reference_date)
            features_list.append(features)

        return pd.DataFrame(features_list)

    def fit_transform(
        self,
        patient_data: pd.DataFrame,
        fill_data: pd.DataFrame,
        reference_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(patient_data, fill_data)
        return self.transform(patient_data, fill_data, reference_date)

    def _engineer_patient_features(
        self, patient: pd.Series, fills: pd.DataFrame, reference_date: date
    ) -> dict[str, Any]:
        """Engineer all features for a single patient."""
        features = {"patient_id": patient["patient_id"]}

        # Add each feature category
        features.update(self._demographic_features(patient))
        features.update(self._refill_history_features(fills, reference_date))
        features.update(self._medication_complexity_features(fills, reference_date))
        features.update(self._cost_features(fills, reference_date))
        features.update(self._clinical_risk_features(patient, fills))
        features.update(self._behavioral_features(patient, fills, reference_date))

        return features

    def _demographic_features(self, patient: pd.Series) -> dict[str, Any]:
        """Extract demographic features."""
        features = {
            "age": patient.get("age", 0),
            "age_bucket": self._get_age_bucket(patient.get("age", 0)),
        }

        # Encode categorical variables
        for col in ["gender", "plan_type"]:
            if col in patient.index:
                value = patient[col] if pd.notna(patient[col]) else "unknown"
                if col in self.label_encoders:
                    try:
                        features[f"{col}_encoded"] = self.label_encoders[col].transform([value])[0]
                    except ValueError:
                        features[f"{col}_encoded"] = -1
                else:
                    features[f"{col}_encoded"] = hash(value) % 100

        return features

    def _get_age_bucket(self, age: int) -> int:
        """Convert age to bucket (0-4)."""
        if age < 30:
            return 0
        elif age < 50:
            return 1
        elif age < 65:
            return 2
        elif age < 75:
            return 3
        else:
            return 4

    def _refill_history_features(
        self, fills: pd.DataFrame, reference_date: date
    ) -> dict[str, float]:
        """Calculate refill history features."""
        features = {
            "total_fills": 0,
            "fills_last_90_days": 0,
            "fills_last_180_days": 0,
            "fills_last_365_days": 0,
            "average_days_between_fills": 0.0,
            "std_days_between_fills": 0.0,
            "gap_count": 0,
            "max_gap_days": 0.0,
            "average_gap_days": 0.0,
            "pdc_90_days": 0.0,
            "pdc_180_days": 0.0,
            "pdc_365_days": 0.0,
            "days_since_last_fill": 999,
            "is_overdue": 0,
            "refill_consistency_score": 0.0,
        }

        if fills.empty:
            return features

        # Convert fill dates
        fills = fills.copy()
        fills["fill_date"] = pd.to_datetime(fills["fill_date"]).dt.date
        fills = fills.sort_values("fill_date")

        # Total fills
        features["total_fills"] = len(fills)

        # Fills in different time windows
        for days, key in [(90, "fills_last_90_days"), (180, "fills_last_180_days"), (365, "fills_last_365_days")]:
            cutoff = reference_date - timedelta(days=days)
            features[key] = len(fills[fills["fill_date"] >= cutoff])

        # Days between fills
        if len(fills) > 1:
            fill_dates = fills["fill_date"].tolist()
            days_between = [
                (fill_dates[i + 1] - fill_dates[i]).days for i in range(len(fill_dates) - 1)
            ]
            features["average_days_between_fills"] = np.mean(days_between)
            features["std_days_between_fills"] = np.std(days_between) if len(days_between) > 1 else 0

        # Gap analysis
        gaps = self._calculate_gaps(fills, reference_date)
        if gaps:
            features["gap_count"] = len(gaps)
            features["max_gap_days"] = max(gaps)
            features["average_gap_days"] = np.mean(gaps)

        # PDC calculations
        for days, key in [(90, "pdc_90_days"), (180, "pdc_180_days"), (365, "pdc_365_days")]:
            features[key] = self._calculate_pdc(fills, reference_date, days)

        # Days since last fill
        if not fills.empty:
            last_fill = fills["fill_date"].max()
            last_days_supply = fills.loc[fills["fill_date"] == last_fill, "days_supply"].iloc[0]
            features["days_since_last_fill"] = (reference_date - last_fill).days
            expected_refill = last_fill + timedelta(days=int(last_days_supply))
            features["is_overdue"] = 1 if reference_date > expected_refill else 0

        # Refill consistency (coefficient of variation)
        if features["average_days_between_fills"] > 0:
            features["refill_consistency_score"] = 1 - min(
                features["std_days_between_fills"] / features["average_days_between_fills"], 1
            )

        return features

    def _calculate_gaps(self, fills: pd.DataFrame, reference_date: date) -> list[int]:
        """Calculate medication gaps (periods without coverage)."""
        if len(fills) < 2:
            return []

        gaps = []
        fills = fills.sort_values("fill_date")

        for i in range(len(fills) - 1):
            current_fill = fills.iloc[i]
            next_fill = fills.iloc[i + 1]

            coverage_end = current_fill["fill_date"] + timedelta(days=int(current_fill["days_supply"]))
            gap_days = (next_fill["fill_date"] - coverage_end).days

            if gap_days > self.config.gap_threshold_days:
                gaps.append(gap_days)

        return gaps

    def _calculate_pdc(
        self, fills: pd.DataFrame, reference_date: date, lookback_days: int
    ) -> float:
        """
        Calculate Proportion of Days Covered (PDC).

        PDC = Days with medication / Total days in period
        """
        if fills.empty:
            return 0.0

        period_start = reference_date - timedelta(days=lookback_days)
        period_fills = fills[fills["fill_date"] >= period_start].copy()

        if period_fills.empty:
            return 0.0

        # Create daily coverage array
        covered_days = set()

        for _, fill in period_fills.iterrows():
            fill_date = fill["fill_date"]
            days_supply = int(fill["days_supply"])

            for day_offset in range(days_supply):
                coverage_date = fill_date + timedelta(days=day_offset)
                if period_start <= coverage_date <= reference_date:
                    covered_days.add(coverage_date)

        pdc = len(covered_days) / lookback_days
        return min(pdc, 1.0)  # Cap at 1.0

    def _medication_complexity_features(
        self, fills: pd.DataFrame, reference_date: date
    ) -> dict[str, Any]:
        """Calculate medication complexity features."""
        features = {
            "unique_medications": 0,
            "unique_medications_active": 0,
            "polypharmacy_flag": 0,
            "average_days_supply": 0.0,
            "multiple_daily_doses": 0,
            "complex_regimen_score": 0.0,
        }

        if fills.empty:
            return features

        # Unique medications
        features["unique_medications"] = fills["medication_ndc"].nunique()

        # Active medications (filled in last 90 days with sufficient supply)
        cutoff = reference_date - timedelta(days=90)
        recent_fills = fills[fills["fill_date"] >= cutoff]
        features["unique_medications_active"] = recent_fills["medication_ndc"].nunique()

        # Polypharmacy (5+ active medications)
        features["polypharmacy_flag"] = 1 if features["unique_medications_active"] >= 5 else 0

        # Average days supply
        features["average_days_supply"] = fills["days_supply"].mean()

        # Complex regimen score (simplified)
        complexity = 0
        complexity += min(features["unique_medications_active"] / 5, 1) * 0.4
        complexity += (1 - min(features["average_days_supply"] / 90, 1)) * 0.3
        complexity += features["polypharmacy_flag"] * 0.3
        features["complex_regimen_score"] = complexity

        return features

    def _cost_features(self, fills: pd.DataFrame, reference_date: date) -> dict[str, float]:
        """Calculate cost-related features."""
        features = {
            "average_copay": 0.0,
            "max_copay": 0.0,
            "total_copay_last_year": 0.0,
            "copay_trend": 0.0,
            "high_cost_medication_flag": 0,
        }

        if fills.empty:
            return features

        features["average_copay"] = fills["copay_amount"].mean()
        features["max_copay"] = fills["copay_amount"].max()

        # Last year costs
        cutoff = reference_date - timedelta(days=365)
        recent_fills = fills[fills["fill_date"] >= cutoff]
        features["total_copay_last_year"] = recent_fills["copay_amount"].sum()

        # Copay trend (comparing first half vs second half of year)
        if len(recent_fills) >= 4:
            mid_point = reference_date - timedelta(days=182)
            first_half = recent_fills[recent_fills["fill_date"] < mid_point]["copay_amount"].mean()
            second_half = recent_fills[recent_fills["fill_date"] >= mid_point]["copay_amount"].mean()
            if first_half > 0:
                features["copay_trend"] = (second_half - first_half) / first_half

        # High cost flag (>$50 copay)
        features["high_cost_medication_flag"] = 1 if features["max_copay"] > 50 else 0

        return features

    def _clinical_risk_features(
        self, patient: pd.Series, fills: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate clinical risk features."""
        features = {
            "has_depression_diagnosis": 0,
            "has_anxiety_diagnosis": 0,
            "has_diabetes_diagnosis": 0,
            "has_hypertension_diagnosis": 0,
            "high_risk_diagnosis_count": 0,
            "chronic_condition_count": 0,
        }

        diagnosis_codes = patient.get("diagnosis_codes", [])
        if isinstance(diagnosis_codes, str):
            diagnosis_codes = diagnosis_codes.split(",")

        for code in diagnosis_codes:
            code = str(code).strip().upper()
            if code.startswith(("F32", "F33")):
                features["has_depression_diagnosis"] = 1
            if code.startswith("F41"):
                features["has_anxiety_diagnosis"] = 1
            if code.startswith("E11"):
                features["has_diabetes_diagnosis"] = 1
            if code.startswith("I10"):
                features["has_hypertension_diagnosis"] = 1

            # Count high-risk diagnoses
            for high_risk_prefix in self.config.high_risk_diagnosis_codes:
                if code.startswith(high_risk_prefix):
                    features["high_risk_diagnosis_count"] += 1
                    break

        # Chronic condition indicator
        features["chronic_condition_count"] = (
            features["has_diabetes_diagnosis"]
            + features["has_hypertension_diagnosis"]
            + features["has_depression_diagnosis"]
            + features["has_anxiety_diagnosis"]
        )

        return features

    def _behavioral_features(
        self, patient: pd.Series, fills: pd.DataFrame, reference_date: date
    ) -> dict[str, Any]:
        """Calculate behavioral/engagement features."""
        features = {
            "has_preferred_channel": 0,
            "mail_order_ratio": 0.0,
            "pharmacy_switches": 0,
            "recent_medication_changes": 0,
            "weekend_fill_ratio": 0.0,
        }

        # Preferred channel
        if pd.notna(patient.get("preferred_channel")):
            features["has_preferred_channel"] = 1

        if fills.empty:
            return features

        # Mail order vs retail (simplified: longer days_supply = likely mail order)
        mail_order_fills = fills[fills["days_supply"] >= 84]
        features["mail_order_ratio"] = len(mail_order_fills) / len(fills)

        # Pharmacy switches
        if "pharmacy_npi" in fills.columns:
            features["pharmacy_switches"] = max(0, fills["pharmacy_npi"].nunique() - 1)

        # Recent medication changes (new medications in last 90 days)
        cutoff = reference_date - timedelta(days=90)
        older_meds = set(fills[fills["fill_date"] < cutoff]["medication_ndc"])
        recent_meds = set(fills[fills["fill_date"] >= cutoff]["medication_ndc"])
        features["recent_medication_changes"] = len(recent_meds - older_meds)

        # Weekend fill ratio
        fills = fills.copy()
        fills["day_of_week"] = pd.to_datetime(fills["fill_date"]).dt.dayofweek
        weekend_fills = fills[fills["day_of_week"] >= 5]
        features["weekend_fill_ratio"] = len(weekend_fills) / len(fills) if len(fills) > 0 else 0

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names (excluding patient_id)."""
        return [
            # Demographic
            "age",
            "age_bucket",
            "gender_encoded",
            "plan_type_encoded",
            # Refill history
            "total_fills",
            "fills_last_90_days",
            "fills_last_180_days",
            "fills_last_365_days",
            "average_days_between_fills",
            "std_days_between_fills",
            "gap_count",
            "max_gap_days",
            "average_gap_days",
            "pdc_90_days",
            "pdc_180_days",
            "pdc_365_days",
            "days_since_last_fill",
            "is_overdue",
            "refill_consistency_score",
            # Medication complexity
            "unique_medications",
            "unique_medications_active",
            "polypharmacy_flag",
            "average_days_supply",
            "multiple_daily_doses",
            "complex_regimen_score",
            # Cost
            "average_copay",
            "max_copay",
            "total_copay_last_year",
            "copay_trend",
            "high_cost_medication_flag",
            # Clinical risk
            "has_depression_diagnosis",
            "has_anxiety_diagnosis",
            "has_diabetes_diagnosis",
            "has_hypertension_diagnosis",
            "high_risk_diagnosis_count",
            "chronic_condition_count",
            # Behavioral
            "has_preferred_channel",
            "mail_order_ratio",
            "pharmacy_switches",
            "recent_medication_changes",
            "weekend_fill_ratio",
        ]

    def get_feature_importance_mapping(self) -> dict[str, str]:
        """Get human-readable descriptions for features."""
        return {
            "pdc_90_days": "Medication coverage in last 90 days",
            "pdc_180_days": "Medication coverage in last 6 months",
            "gap_count": "Number of gaps in medication coverage",
            "max_gap_days": "Longest gap without medication",
            "days_since_last_fill": "Days since last prescription fill",
            "is_overdue": "Currently overdue for refill",
            "average_copay": "Average out-of-pocket cost",
            "high_cost_medication_flag": "Has high-cost medication",
            "has_depression_diagnosis": "Depression diagnosis present",
            "polypharmacy_flag": "Taking 5+ medications",
            "complex_regimen_score": "Medication regimen complexity",
            "refill_consistency_score": "Consistency of refill timing",
            "age": "Patient age",
            "chronic_condition_count": "Number of chronic conditions",
        }
