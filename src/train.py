from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import joblib
from clustering import apply_clustering

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate():
    print("--- Étape 5 : Classification Supervisée et Pipeline ---")
    df = apply_clustering()
    
    # Définition de y et X
    y = df['risk_category']
    X = df.drop(columns=['Cluster', 'risk_category','id'])
    
    # Division Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Gestion du déséquilibre des classes (Oversampling)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    # Comparaison de plusieurs modèles de classification
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        print(f"\n--- {name} ---")
        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, pos_label=1))
        print("Recall   :", recall_score(y_test, y_pred, pos_label=1))
        print("F1-score :", f1_score(y_test, y_pred, pos_label=1))

    # Exemple d'optimisation (Hyperparameter Tuning avec GridSearchCV pour Logistic Regression)
    param_grid_lr = {
        "model__C": [0.01, 0.1, 1, 10]
    }
    
    # Construction du Pipeline final (incluant Imputation, Scaling et Modèle)
    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # GridSearch sur le pipeline
    grid = GridSearchCV(pipeline, param_grid_lr, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print("\nMeilleurs paramètres trouvés :", grid.best_params_)
    best_pipeline = grid.best_estimator_
    
    # Évaluation du meilleur modèle sur le jeu de test
    y_pred_best = best_pipeline.predict(X_test)
    print("Recall final après Tuning :", recall_score(y_test, y_pred_best, pos_label=1))
    
    # Sauvegarde du Pipeline final
    model_path = MODELS_DIR / "diabetes_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nPipeline sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    train_and_evaluate()