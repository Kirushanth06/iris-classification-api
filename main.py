from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
app = FastAPI(title="Iris Classification API - Kirushanth")

species = ["setosa", "versicolor", "virginica"]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    try:
        features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features).max()
        return PredictionOutput(prediction=species[pred], confidence=float(prob))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": species
    }
