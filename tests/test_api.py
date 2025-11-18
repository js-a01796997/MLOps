"""Tests for FastAPI endpoints - v2 API with MLflow integration

These tests validate the API endpoints for model inference.
NOTE: Some tests require MLflow server to be running and models to be registered.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

pytestmark = pytest.mark.api

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "api"))

from fastapi.testclient import TestClient
from api_v2 import api_v2, parse_model_id, model_cache, model_cache_metadata
from common import PredictRequest, PredictResponse, validate_predict_input
from fastapi import FastAPI, HTTPException

# Create test app
app = FastAPI()
app.include_router(api_v2)
client = TestClient(app)


class TestCommonValidation:
    """Tests for common validation functions"""

    def test_validate_predict_input_valid_data(self):
        """Test validation passes with valid input"""
        req = PredictRequest(features=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Should not raise exception
        validate_predict_input(req, "test_model", expected_n_features=3)

    def test_validate_predict_input_no_features(self):
        """Test validation fails when features is None"""
        req = PredictRequest(features=None)

        with pytest.raises(HTTPException) as exc_info:
            validate_predict_input(req, "test_model", expected_n_features=3)

        assert exc_info.value.status_code == 400
        assert "features" in str(exc_info.value.detail).lower()

    def test_validate_predict_input_empty_features(self):
        """Test validation fails when features is empty"""
        req = PredictRequest(features=[])

        with pytest.raises(HTTPException) as exc_info:
            validate_predict_input(req, "test_model", expected_n_features=3)

        assert exc_info.value.status_code == 422

    def test_validate_predict_input_inconsistent_row_lengths(self):
        """Test validation fails when rows have different lengths"""
        req = PredictRequest(features=[[1.0, 2.0], [3.0, 4.0, 5.0]])

        with pytest.raises(HTTPException) as exc_info:
            validate_predict_input(req, "test_model", expected_n_features=3)

        assert exc_info.value.status_code == 422
        assert "longitud" in str(exc_info.value.detail).lower() or "length" in str(exc_info.value.detail).lower()

    def test_validate_predict_input_wrong_feature_count(self):
        """Test validation fails when feature count doesn't match expected"""
        req = PredictRequest(features=[[1.0, 2.0]])

        with pytest.raises(HTTPException) as exc_info:
            validate_predict_input(req, "test_model", expected_n_features=3)

        assert exc_info.value.status_code == 422
        assert "features" in str(exc_info.value.detail).lower()

    def test_validate_predict_input_no_expected_count(self):
        """Test validation passes when expected_n_features is None"""
        req = PredictRequest(features=[[1.0, 2.0]])

        # Should not raise exception when expected_n_features is None
        validate_predict_input(req, "test_model", expected_n_features=None)


class TestParseModelId:
    """Tests for parse_model_id function"""

    def test_parse_model_id_with_version(self):
        """Test parsing model_id with numeric version"""
        result = parse_model_id("bike_sharing_xgboost:8")

        assert result == "models:/bike_sharing_xgboost/8"

    def test_parse_model_id_with_stage(self):
        """Test parsing model_id with stage name"""
        result = parse_model_id("bike_sharing_xgboost:Production")

        assert result == "models:/bike_sharing_xgboost/Production"

    def test_parse_model_id_invalid_format_no_colon(self):
        """Test that invalid format (no colon) raises exception"""
        with pytest.raises(HTTPException) as exc_info:
            parse_model_id("bike_sharing_xgboost")

        assert exc_info.value.status_code == 400
        assert "formato" in str(exc_info.value.detail).lower() or "format" in str(exc_info.value.detail).lower()

    def test_parse_model_id_invalid_format_multiple_colons(self):
        """Test that invalid format (multiple colons) raises exception"""
        with pytest.raises(HTTPException) as exc_info:
            parse_model_id("bike:sharing:xgboost:8")

        assert exc_info.value.status_code == 400

    def test_parse_model_id_empty_parts(self):
        """Test that empty name or version raises exception"""
        with pytest.raises(HTTPException) as exc_info:
            parse_model_id(":8")

        assert exc_info.value.status_code == 400


class TestAPIEndpointsWithMocking:
    """Tests for API endpoints using mocking (don't require MLflow server)"""

    def setup_method(self):
        """Clear model cache before each test"""
        model_cache.clear()
        model_cache_metadata.clear()

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/v2/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["api_version"] == "v2"

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_predict_endpoint_success(self, mock_load_model):
        """Test successful prediction with mocked model"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([100.0, 200.0])

        # Mock get_raw_model
        raw_model = Mock()
        raw_model.n_features_in_ = 3
        raw_model.feature_names_in_ = ['feature1', 'feature2', 'feature3']
        mock_model.get_raw_model.return_value = raw_model

        mock_load_model.return_value = mock_model

        # Make request
        response = client.post(
            "/v2/models/test_model:1/predict",
            json={"features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0] == 100.0
        assert data["predictions"][1] == 200.0

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_predict_endpoint_invalid_features(self, mock_load_model):
        """Test prediction with invalid feature count"""
        # Mock model
        mock_model = Mock()
        raw_model = Mock()
        raw_model.n_features_in_ = 5  # Expects 5 features
        raw_model.feature_names_in_ = ['f1', 'f2', 'f3', 'f4', 'f5']
        mock_model.get_raw_model.return_value = raw_model
        mock_load_model.return_value = mock_model

        # Send only 3 features
        response = client.post(
            "/v2/models/test_model:1/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )

        assert response.status_code == 422
        assert "features" in response.json()["detail"].lower()

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_predict_endpoint_model_not_found(self, mock_load_model):
        """Test prediction with non-existent model"""
        mock_load_model.side_effect = Exception("Model not found")

        response = client.post(
            "/v2/models/nonexistent_model:1/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )

        assert response.status_code == 404
        assert "no se pudo cargar" in response.json()["detail"].lower() or "not found" in response.json()["detail"].lower()

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_model_info_endpoint(self, mock_load_model):
        """Test model info endpoint"""
        # Mock model
        mock_model = Mock()
        raw_model = Mock()
        raw_model.n_features_in_ = 10
        raw_model.feature_names_in_ = [f'feature_{i}' for i in range(10)]
        mock_model.get_raw_model.return_value = raw_model
        mock_load_model.return_value = mock_model

        response = client.get("/v2/models/test_model:1/info")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test_model:1"
        assert data["n_features"] == 10
        assert data["mlflow_uri"] == "models:/test_model/1"
        assert "loaded_at" in data

    @patch('api_v2.MlflowClient')
    def test_list_models_endpoint(self, mock_client_class):
        """Test list models endpoint"""
        # Mock MLflow client
        mock_client = Mock()

        # Create mock registered model
        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.creation_timestamp = 1000000
        mock_model.last_updated_timestamp = 2000000

        # Create mock version
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.creation_timestamp = 1000000
        mock_version.last_updated_timestamp = 2000000
        mock_version.run_id = "test_run_id"

        mock_model.latest_versions = [mock_version]
        mock_client.search_registered_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        response = client.get("/v2/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "test_model"
        assert len(data[0]["versions"]) == 1
        assert data[0]["versions"][0]["version"] == "1"


class TestAPIInputValidation:
    """Tests for API input validation"""

    def test_predict_request_schema_valid(self):
        """Test that valid request matches schema"""
        request_data = {"features": [[1.0, 2.0, 3.0]]}
        request = PredictRequest(**request_data)

        assert request.features == [[1.0, 2.0, 3.0]]

    def test_predict_request_schema_allows_none(self):
        """Test that features can be None (will be caught by validation)"""
        request = PredictRequest(features=None)

        assert request.features is None

    def test_predict_response_schema(self):
        """Test PredictResponse schema"""
        response = PredictResponse(predictions=[1.0, 2.0, 3.0])

        assert response.predictions == [1.0, 2.0, 3.0]

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_api_handles_empty_features_list(self, mock_load_model):
        """Test API rejects empty features list"""
        mock_model = Mock()
        raw_model = Mock()
        raw_model.n_features_in_ = 3
        mock_model.get_raw_model.return_value = raw_model
        mock_load_model.return_value = mock_model

        response = client.post(
            "/v2/models/test_model:1/predict",
            json={"features": []}
        )

        assert response.status_code == 422


class TestModelCaching:
    """Tests for model caching functionality"""

    def setup_method(self):
        """Clear model cache before each test"""
        model_cache.clear()
        model_cache_metadata.clear()

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_model_cached_after_first_load(self, mock_load_model):
        """Test that model is cached after first load"""
        mock_model = Mock()
        raw_model = Mock()
        raw_model.n_features_in_ = 3
        raw_model.feature_names_in_ = ['f1', 'f2', 'f3']
        mock_model.get_raw_model.return_value = raw_model
        mock_model.predict.return_value = np.array([100.0])
        mock_load_model.return_value = mock_model

        # First request - should load from MLflow
        client.post(
            "/v2/models/test_model:1/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )

        assert mock_load_model.call_count == 1
        assert "test_model:1" in model_cache

        # Second request - should use cache
        client.post(
            "/v2/models/test_model:1/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )

        # Should still be 1 (not loaded again)
        assert mock_load_model.call_count == 1

    @patch('api_v2.mlflow.pyfunc.load_model')
    def test_different_models_cached_separately(self, mock_load_model):
        """Test that different models are cached separately"""
        def create_mock_model(name):
            mock_model = Mock()
            raw_model = Mock()
            raw_model.n_features_in_ = 3
            raw_model.feature_names_in_ = ['f1', 'f2', 'f3']
            mock_model.get_raw_model.return_value = raw_model
            mock_model.predict.return_value = np.array([100.0])
            return mock_model

        mock_load_model.side_effect = lambda uri: create_mock_model(uri)

        # Load two different models
        client.post(
            "/v2/models/model1:1/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )
        client.post(
            "/v2/models/model2:1/predict",
            json={"features": [[1.0, 2.0, 3.0]]}
        )

        assert "model1:1" in model_cache
        assert "model2:1" in model_cache
        assert mock_load_model.call_count == 2


# Tests that require actual MLflow server (marked for optional execution)
@pytest.mark.skipif(
    True,  # Change to False if you want to run these tests with a real MLflow server
    reason="Requires MLflow server and registered models"
)
class TestAPIWithRealMLflow:
    """Tests that require actual MLflow server connection

    These tests are skipped by default. To run them:
    1. Start MLflow server
    2. Register test models
    3. Change skipif condition to False
    """

    def test_real_model_prediction(self):
        """Test prediction with real model (requires MLflow)"""
        response = client.post(
            "/v2/models/bike_sharing_xgboost:8/predict",
            json={"features": [[0.5] * 10]}  # Adjust feature count as needed
        )

        assert response.status_code == 200
        assert "predictions" in response.json()

    def test_real_model_info(self):
        """Test model info with real model (requires MLflow)"""
        response = client.get("/v2/models/bike_sharing_xgboost:8/info")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "bike_sharing_xgboost:8"

    def test_list_real_models(self):
        """Test listing real models (requires MLflow)"""
        response = client.get("/v2/models")

        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) > 0
