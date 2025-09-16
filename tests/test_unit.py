import os
import sys
import json
import pytest
import pandas as pd
import tempfile
import requests
import time

# Define PROJECT_ROOT first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ensure project root is importable when someone runs the file directly.
sys.path.insert(0, PROJECT_ROOT)

# Import app after ensuring project root is on sys.path
try:
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to using requests if TestClient doesn't work
    client = None

def is_server_running():
    """Check if the server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def test_full_prediction_pipeline_with_real_data():
    """
    Integration test that uses the actual test dataset to validate the full prediction pipeline:
    1. Upload the test CSV file to the /predict endpoint
    2. Download the predictions file
    3. Use one of the predictions to test the /explain endpoint
    """
    # Check if server is running
    if not is_server_running():
        pytest.skip("Server is not running. Please start the server with: uvicorn app.main:app --reload")
        return
    
    # Path to your test dataset
    test_data_path = os.path.join(PROJECT_ROOT, 'data', 'cricket_dataset_test.csv')
    
    # Verify the test file exists
    if not os.path.exists(test_data_path):
        pytest.skip(f"Test data file not found at {test_data_path}")
        return
    
    # Use requests with the running server
    test_with_requests(test_data_path)

def test_with_requests(test_data_path):
    """
    Alternative test using requests when TestClient is not available
    """
    # Wait for server to start
    time.sleep(2)
    
    # Step 1: Upload the file to the predict endpoint
    with open(test_data_path, 'rb') as f:
        files = {'file': ('cricket_dataset_test.csv', f, 'text/csv')}
        response = requests.post("http://localhost:8000/predict", files=files)
    
    # Check the response
    if response.status_code != 200:
        # If we get an error, check if it's due to model loading
        error_detail = response.json().get("detail", "")
        if "model" in error_detail.lower():
            pytest.skip(f"Model loading error: {error_detail}")
            return
        else:
            assert response.status_code == 200, f"Unexpected error: {error_detail}"
    
    response_data = response.json()
    
    # Check response structure
    assert response_data["status"] == "success"
    assert "predictions_file" in response_data
    assert "metadata" in response_data
    
    # Check metadata
    metadata = response_data["metadata"]
    assert metadata["total_rows"] > 0
    
    # Step 2: Download the predictions file
    predictions_file = response_data["predictions_file"]
    predictions_file_path = os.path.join(PROJECT_ROOT, predictions_file)
    
    # Wait a moment for the file to be created
    time.sleep(1)
    
    assert os.path.exists(predictions_file_path)
    
    # Read the predictions file
    predictions_df = pd.read_csv(predictions_file_path)
    assert "prediction" in predictions_df.columns
    assert "prediction_confidence" in predictions_df.columns
    assert len(predictions_df) == metadata["predictions_made"]
    
    # Step 3: Use the first prediction to test the explain endpoint
    if len(predictions_df) > 0:
        first_prediction = predictions_df.iloc[0]
        
        # Prepare data for the explain endpoint
        explain_data = {
            "prediction": int(first_prediction['prediction']),
            "confidence": float(first_prediction['prediction_confidence']),
            "total_runs": int(first_prediction['total_runs']),
            "wickets": int(first_prediction['wickets']),
            "target": int(first_prediction['target']),
            "balls_left": int(first_prediction['balls_left'])
        }
        
        # Call the explain endpoint
        explain_response = requests.post("http://localhost:8000/explain/1", json=explain_data)
        
        # Check the explain response
        assert explain_response.status_code == 200
        explain_result = explain_response.json()
        
        assert "prediction" in explain_result
        assert "confidence" in explain_result
        assert "explanation" in explain_result
        assert isinstance(explain_result["explanation"], str)
        assert len(explain_result["explanation"]) > 0
        
        print(f"Prediction: {explain_result['prediction']}")
        print(f"Confidence: {explain_result['confidence']}")
        print(f"Explanation: {explain_result['explanation']}")
    
    # Clean up predictions file
    if os.path.exists(predictions_file_path):
        os.remove(predictions_file_path)

if __name__ == "__main__":
    # Run the test if executed directly
    test_full_prediction_pipeline_with_real_data()