"""
Tests for the risk prediction model.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from src.models.risk_predictor import AdherenceRiskPredictor, RiskPrediction
from src.models.schemas import RiskLevel


@pytest.fixture
def predictor():
    """Create a predictor instance."""
    return AdherenceRiskPredictor()


@pytest.fixture
def sample_patient_data():
    """Create sample patient data."""
    return pd.DataFrame([
        {
            "patient_id": "P001",
            "age": 55,
            "gender": "M",
            "plan_type": "commercial",
            "diagnosis_codes": ["E11.9", "I10"],
            "preferred_channel": "sms",
        }
    ])


@pytest.fixture
def sample_fills_data():
    """Create sample medication fill data."""
    today = date.today()
    return pd.DataFrame([
        {
            "fill_id": "F001",
            "patient_id": "P001",
            "medication_ndc": "00093-7212-01",
            "medication_name": "Metformin 500mg",
            "fill_date": today - timedelta(days=60),
            "days_supply": 30,
            "refill_number": 0,
            "copay_amount": 10.00,
            "quantity": 30,
        },
        {
            "fill_id": "F002",
            "patient_id": "P001",
            "medication_ndc": "00093-7212-01",
            "medication_name": "Metformin 500mg",
            "fill_date": today - timedelta(days=25),
            "days_supply": 30,
            "refill_number": 1,
            "copay_amount": 10.00,
            "quantity": 30,
        },
    ])


class TestAdherenceRiskPredictor:
    """Tests for AdherenceRiskPredictor."""

    def test_predict_returns_predictions(
        self, predictor, sample_patient_data, sample_fills_data
    ):
        """Test that predict returns a list of predictions."""
        predictions = predictor.predict(
            sample_patient_data,
            sample_fills_data,
            prediction_horizon_days=30,
        )

        assert len(predictions) == 1
        assert isinstance(predictions[0], RiskPrediction)

    def test_prediction_has_required_fields(
        self, predictor, sample_patient_data, sample_fills_data
    ):
        """Test that predictions have all required fields."""
        predictions = predictor.predict(
            sample_patient_data,
            sample_fills_data,
        )

        pred = predictions[0]
        assert pred.patient_id == "P001"
        assert 0 <= pred.risk_score <= 100
        assert isinstance(pred.risk_level, RiskLevel)
        assert len(pred.top_factors) <= 3
        assert 0 <= pred.confidence <= 1

    def test_risk_level_categorization(self, predictor):
        """Test risk level is correctly assigned based on score."""
        assert predictor._get_risk_level(25) == RiskLevel.LOW
        assert predictor._get_risk_level(50) == RiskLevel.MEDIUM
        assert predictor._get_risk_level(85) == RiskLevel.HIGH

    def test_predict_single_convenience_method(self, predictor):
        """Test the predict_single convenience method."""
        prediction = predictor.predict_single(
            patient_id="P002",
            patient_data={
                "age": 45,
                "gender": "F",
                "plan_type": "medicare",
                "diagnosis_codes": ["F32.9"],
            },
            fills=[
                {
                    "fill_id": "F100",
                    "patient_id": "P002",
                    "medication_ndc": "00093-4561-01",
                    "medication_name": "Sertraline 50mg",
                    "fill_date": date.today() - timedelta(days=45),
                    "days_supply": 30,
                    "refill_number": 0,
                    "copay_amount": 15.00,
                    "quantity": 30,
                }
            ],
        )

        assert prediction.patient_id == "P002"
        assert prediction.risk_score >= 0

    def test_empty_fills_returns_prediction(self, predictor, sample_patient_data):
        """Test that prediction works even with no fill history."""
        empty_fills = pd.DataFrame()

        predictions = predictor.predict(
            sample_patient_data,
            empty_fills,
        )

        assert len(predictions) == 1
        # No history should indicate higher risk
        assert predictions[0].risk_score > 30

    def test_prediction_to_response(
        self, predictor, sample_patient_data, sample_fills_data
    ):
        """Test conversion to API response format."""
        predictions = predictor.predict(
            sample_patient_data,
            sample_fills_data,
        )

        response = predictions[0].to_response()

        assert response.patient_id == "P001"
        assert response.predicted_risk_window_start == date.today()
        assert response.model_version == predictor.MODEL_VERSION

    def test_model_info(self, predictor):
        """Test model info retrieval."""
        info = predictor.get_model_info()

        assert "version" in info
        assert "is_fitted" in info
        assert "risk_thresholds" in info


class TestFeatureEngineer:
    """Tests for feature engineering."""

    def test_pdc_calculation(self, predictor, sample_patient_data, sample_fills_data):
        """Test PDC is calculated correctly."""
        from src.models.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        features = fe.fit_transform(sample_patient_data, sample_fills_data)

        # Should have PDC features
        assert "pdc_90_days" in features.columns
        # With fills covering 30 days in a 90 day period, PDC should be ~0.33-0.67
        pdc = features.iloc[0]["pdc_90_days"]
        assert 0 <= pdc <= 1

    def test_feature_count(self, predictor, sample_patient_data, sample_fills_data):
        """Test that expected number of features are generated."""
        from src.models.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        features = fe.fit_transform(sample_patient_data, sample_fills_data)

        # Should have 40+ features
        assert len(features.columns) >= 30
