from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Medical Risk Prediction API",
    description="API d'inférence pour la prédiction du risque de diabète connectée au Model Registry de MLflow",
    version="1.0.0"
)

# Configuration de l'URI de suivi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = "MedicalRiskPipelineModel"

def load_production_model():
    """Charge le modèle directement depuis le Model Registry de MLflow au stage Production"""
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        print(f"Chargement du modèle depuis l'URI : {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis MLflow Registry : {e}")
        return None

# Schéma Pydantic pour valider les données entrantes des constantes cliniques
class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction du risque de diabète (MLflow Production Model)"}

@app.post("/predict")
def predict_risk(data: PatientData):
    # Chargement dynamique du modèle de production
    model = load_production_model()
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Le modèle en production n'a pas pu être chargé depuis MLflow Registry."
        )
    
    # Transformation des données reçues en DataFrame respectant l'ordre des features
    input_df = pd.DataFrame([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    try:
        # Réalisation de la prédiction et extraction de la probabilité
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else None
        
        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        
        return {
            "prediction": prediction,
            "risk_category": risk_label,
            "probability": probability,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du calcul de la prédiction : {str(e)}")