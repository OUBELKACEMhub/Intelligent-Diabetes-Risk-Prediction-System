from pathlib import Path
import pandas as pd
import hashlib
import importlib.metadata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn

# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def run_classification_and_registry():
    # 1. Configuration MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = "Medical_Risk_Classification"
    mlflow.set_experiment(experiment_name)
    
    print(f"--- Connecté à MLflow sur l'expérience : {experiment_name} ---")
    
    # 2. Chargement des données (depuis le fichier enrichi par le clustering)
    df = pd.read_csv(DATA_DIR / "processed" / "pima_diabetes_clusters_risk.csv")    
    X = df.drop(columns=["risk_category", "Cluster", "id"], errors="ignore")    
    y = df["risk_category"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparamètres pour la Logistic Regression
    C = 1.0
    random_state = 42
    
    with mlflow.start_run(run_name="Classification_LR_Production_Run") as run:
        
        # --- A. LOGGING DES HYPERPARAMÈTRES & TRAÇABILITÉ ---
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("random_state", random_state)
        
        features_list = list(X.columns)
        mlflow.log_text(str(features_list), "features_used.txt")
        mlflow.log_param("num_features", len(features_list))
        
        # Versions des bibliothèques
        for lib in ["mlflow", "scikit-learn", "pandas", "numpy"]:
            try:
                version = importlib.metadata.version(lib)
                mlflow.log_param(f"lib_{lib}_version", version)
            except Exception:
                pass
                
        # Empreinte du dataset
        dataset_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("dataset_shape", str(df.shape))
        
        # --- B. ENTRAÎNEMENT & MÉTRIQUES ---
        clf = LogisticRegression(C=C, random_state=random_state)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        cm_image_path = MODELS_DIR / "confusion_matrix.png"
        plt.savefig(cm_image_path)
        plt.close()
        mlflow.log_artifact(str(cm_image_path))
        
        # --- C. ENREGISTREMENT DANS LE MODEL REGISTRY ---
        signature = infer_signature(X_train, y_pred)
        model_name = "MedicalRiskPipelineModel"
        
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="classification_model",
            signature=signature,
            registered_model_name=model_name
        )
        
    # --- D. TRANSITION VERS LE STAGE PRODUCTION ---
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
    
    if latest_versions:
        version_to_promote = latest_versions[-1].version
        client.transition_model_version_stage(
            name=model_name,
            version=version_to_promote,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Succès : Le modèle '{model_name}' (Version {version_to_promote}) a été basculé vers le stage Production !")

if __name__ == "__main__":
    run_classification_and_registry()