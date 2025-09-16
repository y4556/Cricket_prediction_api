# Cricket Match Prediction API

A FastAPI application that predicts T20 cricket match outcomes and provides AI-powered explanations using Groq.

## Features

- Predict match outcomes based on current game state
- Filter input data based on specific criteria (balls_left < 60, target > 120)
- Generate human-readable explanations using Groq AI
- Comprehensive error handling and logging
- Feature engineering (required run rate, wickets remaining)


## API Endpoints
1. POST /predict
Upload a CSV file for predictions. The CSV must include columns: total_runs, wickets, target, balls_left.

2. POST /explain/{prediction_id}
Get an explanation for a prediction. Requires a JSON body with prediction details.

3. GET /health
Health check endpoint.

4. GET /predictions/{filename}
Download a prediction results file.

## Model Performance
The model (Random Forest) was trained on the cricket dataset and achieved the following performance:
Accuracy: 85.2%

Precision: 0.86

Recall: 0.84

F1-Score: 0.85


## Setup
```bash
1. Install dependencies:

pip install -r requirements.txt

2. Set up environment variables in .env:

env
GROQ_API_KEY=your_groq_api_key_here
MODEL_PATH=models/trained_model.pkl

3. Place your trained model in the models/ directory

4. Run the application:

uvicorn app.main:app --reload

