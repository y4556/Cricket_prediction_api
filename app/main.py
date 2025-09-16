import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.testclient import TestClient
import os
from typing import List, Dict, Any
from datetime import datetime
import pickle
from dotenv import load_dotenv

# Import utility functions
from app.utils import (
    save_upload_file, 
    cleanup_file, 
    generate_explanation,
    validate_columns,
    preprocess_data
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cricket Prediction API", version="1.0.0")

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Load the trained model at application startup"""
    global model
    model_path = os.getenv("MODEL_PATH", "models/random_forest.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.post("/predict")
async def predict_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Predict match outcomes from uploaded CSV file
    """
    try:
        # Save uploaded file temporarily
        file_path = await save_upload_file(file)
        background_tasks.add_task(cleanup_file, file_path)
        
        # Read and validate CSV
        df = pd.read_csv(file_path)
        validate_columns(df)
        
        # Apply filtering (balls_left < 60 and target > 120)
        filtered_df = df[(df['balls_left'] < 60) & (df['target'] > 120)].copy()
        
        if len(filtered_df) == 0:
            raise HTTPException(status_code=400, detail="No rows match the filtering criteria")
        
        # Preprocess data with feature engineering
        processed_data = preprocess_data(filtered_df)
        
        # Make predictions
        predictions = model.predict(processed_data[['total_runs', 'wickets', 'target', 'balls_left', 
                                                   'required_run_rate', 'wickets_remaining']])
        
        # Get prediction probabilities for confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data[['total_runs', 'wickets', 'target', 'balls_left', 
                                                              'required_run_rate', 'wickets_remaining']])
            confidences = [max(prob) for prob in probabilities]
        else:
            confidences = [0.5] * len(predictions)
        
        # Save results
        results_df = filtered_df.copy()
        results_df['prediction'] = predictions
        results_df['prediction_confidence'] = confidences
        
        output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "predictions_file": output_path,
            "metadata": {
                "total_rows": len(df),
                "filtered_rows": len(filtered_df),
                "predictions_made": len(predictions),
                "model_used": os.path.basename(os.getenv("MODEL_PATH", "random_forest.pkl"))
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predictions/{filename}")
async def download_predictions(filename: str):
    """
    Download prediction results file
    """
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filename, media_type="text/csv")

@app.post("/explain/{prediction_id}")
async def explain_prediction(prediction_id: int, data: Dict[str, Any]):
    """
    Generate human-readable explanation for a prediction
    """
    try:
        explanation = await generate_explanation(data, prediction_id)
        return {
            "prediction": data.get('prediction', 0),
            "confidence": data.get('confidence', 0.5),
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Explanation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))