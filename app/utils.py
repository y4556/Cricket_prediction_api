import os
import aiofiles
import json
from fastapi import HTTPException
import logging
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import numpy as np

load_dotenv()
logger = logging.getLogger(__name__)

# Required columns from the dataset
REQUIRED_COLUMNS = ['total_runs', 'wickets', 'target', 'balls_left']

def validate_columns(df: pd.DataFrame):
    """Validate that CSV contains required columns"""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required columns: {', '.join(missing_columns)}"
        )

def preprocess_data(df: pd.DataFrame):
    """Preprocess data for model prediction with feature engineering"""
    # Create a copy to avoid modifying the original
    processed = df.copy()
    
    # Feature engineering (same as in your training)
    processed['required_run_rate'] = (processed['target'] - processed['total_runs']) / (processed['balls_left'] / 6)
    processed['required_run_rate'] = processed['required_run_rate'].replace([np.inf, -np.inf], 0)
    processed['required_run_rate'] = processed['required_run_rate'].fillna(0)
    
    processed['wickets_remaining'] = 10 - processed['wickets']
    
    return processed

async def save_upload_file(upload_file) -> str:
    """Save uploaded file to temporary location"""
    file_path = f"temp_{upload_file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    return file_path

def cleanup_file(file_path: str):
    """Remove temporary file"""
    if os.path.exists(file_path):
        os.remove(file_path)

async def generate_explanation(data: dict, prediction_id: int) -> str:
    """
    Generate explanation using Groq API
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        logger.warning("Groq API key not found, using mock explanation")
        return generate_mock_explanation(data, prediction_id)
    
    # Prepare prompt based on prediction data
    prompt = create_explanation_prompt(data, prediction_id)
    
    try:
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Create completion
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="high",
            stream=False
        )
        
        return completion.choices[0].message.content
                    
    except Exception as e:
        logger.error(f"Groq API call failed: {str(e)}")
        return generate_mock_explanation(data, prediction_id)

def create_explanation_prompt(data: dict, prediction_id: int) -> str:
    """
    Create prompt for explanation generation
    """
    prediction = data.get('prediction', 0)
    confidence = data.get('confidence', 0.5)
    
    prompt_template = f"""
    You are a cricket analyst explaining the prediction of a machine learning model for a T20 cricket match.
    
    Match situation:
    - Total runs scored so far: {data.get('total_runs', 0)}
    - Wickets fallen: {data.get('wickets', 0)}
    - Target runs to win: {data.get('target', 0)}
    - Balls left: {data.get('balls_left', 0)}
    - Required run rate: {data.get('required_run_rate', 0):.2f}
    - Wickets remaining: {data.get('wickets_remaining', 10)}
    
    The model predicts: {'WIN' if prediction == 1 else 'LOSS'} with {confidence:.2f} confidence.
    
    Provide a concise, insightful explanation of this prediction considering:
    1. The current match situation
    2. Required run rate vs current run rate
    3. Wickets in hand and balls remaining
    4. Historical success rates in similar situations
    
    Keep the explanation under 5 sentences and suitable for cricket fans.
    """
    
    return prompt_template

def generate_mock_explanation(data: dict, prediction_id: int) -> str:
    """
    Generate a mock explanation when Groq API is unavailable
    """
    prediction = data.get('prediction', 0)
    
    if prediction == 1:
        return "Based on the current match situation, the chasing team has a good chance of winning because they have enough wickets in hand and the required run rate is achievable with the number of balls remaining."
    else:
        return "The chasing team faces a difficult situation as the required run rate is high with limited wickets remaining, making a successful chase unlikely based on historical data."