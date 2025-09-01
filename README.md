# ML Pipeline 

# Project Architecture 
project/
│── Dockerfile # Docker file for serving endpoint
│── requirements.txt # Instalation requirements for serving endpoint
│── slice_analysis.py # Indevidual case analysis
│── patient_input.json # Input JSON file for indevidual case analysis
│── app.py # serving endpoint run file
│── likely_error_causes.txt 
│── README.md
│── training/
│     ├── Dockerfile
│     ├── config.yaml
│     ├── requirements.txt
│     ├── data_gen.py
│     ├── data_prep.py
│     ├── train.py
│     ├── data/patient_year_data.parquet # Generated data directory
│     ├── artifacts/ # Trained model and results directory

# Work flow

# Enter /training directory

## Prepare data
!python data_gen.py
Place your patient-year Parquet at /path/to/data/patient_year_data.parquet

## Build Docker image
docker build -t clinical-lgbm .

## Run container
docker run --rm -v /path/to/data:/data -v /path/to/artifacts:/artifacts -it clinical-pipeline:latest

The container reads /data/patient_year_data.parquet ,
and writes artifacts to /artifacts/lgbm_model.joblib

# Go up directory

## Build Docker image
docker build -t clinical-predictor .

## Run container
docker run -p 8000:8000 clinical-predictor

## Request to serving endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @patient_input.json


## Example response
{
  "predicted_probability": 0.34,
  "predicted_class": 1
}
