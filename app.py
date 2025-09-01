import mlflow
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Load model (logged in MLflow)
model_path = "/training/artifacts/lgbm_model.joblib"  # update with MLflow run
model = joblib.load(model_path)

# Define expected input schema
class PatientInput(BaseModel):
    features: dict  # key:value for patient features

app = FastAPI(title="Clinical Risk Prediction API")

@app.post("/predict")
def predict(input_data: PatientInput):
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data.features])

    preproc_path = "/training/artifacts/preprocessor.joblib"
    preproc = joblib.load(preproc_path)
     # Transform datasets (dense arrays)
    X_test = preproc.transform(df)

    # Predict probability
    prob = model.predict_proba(X_test)[:,1][0]
    pred_class = int(prob > 0.25)  # use tuned threshold (adjust as needed)

    return {
        "probability": float(prob),
        "predicted_class": pred_class
    }
