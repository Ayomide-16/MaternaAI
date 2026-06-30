# MaternaAI - Inference API

MaternaAI is a Flask-based inference API designed to predict maternal health risk levels. It serves a trained XGBoost machine learning model and provides highly interpretable predictions using SHAP (SHapley Additive exPlanations).

## Overview

The API accepts patient data, scales it according to the pre-trained scaler, and uses the XGBoost model to classify the maternal health risk into one of three categories:
- **Low Risk**
- **Medium Risk**
- **High Risk**

In addition to the prediction, the API uses SHAP to analyze the specific factors contributing to the risk level for that individual patient. It returns the top 3 contributing factors along with tailored clinical recommendations.

## Prerequisites

Ensure you have Python 3 installed. The required packages are listed in `requirements.txt`.

- Flask
- Flask-CORS
- XGBoost
- scikit-learn
- SHAP
- pandas & numpy
- gunicorn (for production deployment)

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ayomide-16/MaternaAI.git
   cd MaternaAI
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the following model artifacts are present in the root directory:
   - `model.json`: The trained XGBoost model.
   - `scaler.joblib`: The pre-trained scikit-learn scaler.
   - `feature_cols.json`: The list of expected features.
   - `feature_importance.json`: Global feature importance data.

## Running the Application

To run the API locally in development mode:

```bash
python app.py
```
The server will start on `http://0.0.0.0:5000`.

## API Endpoints

### 1. Health Check
`GET /health`

Returns a simple status to verify the API is up and running.

**Response:**
```json
{
  "status": "ok"
}
```

### 2. Predict Risk
`POST /predict`

Accepts patient data and returns the risk level, confidence score, and explainability factors.

**Request Body:**
A JSON object containing all the required features defined in `feature_cols.json`.

```json
{
  "Age": 25,
  "SystolicBP": 120,
  "DiastolicBP": 80,
  "BS": 7.5,
  "BodyTemp": 98.0,
  "HeartRate": 70
}
```
*(Note: Replace with actual features required by your model).*

**Response:**
```json
{
  "confidence": 0.892,
  "recommendation": "Routine follow-up at next scheduled antenatal visit.",
  "risk_class": 0,
  "risk_label": "Low",
  "top_factors": [
    {
      "feature": "SystolicBP",
      "impact": "decreases risk",
      "magnitude": 0.45,
      "value": 120
    },
    ...
  ]
}
```

## Deployment

For production, it is recommended to use a WSGI HTTP Server like `gunicorn`.
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
You can deploy this API to cloud platforms like Render, Railway, or Heroku.
