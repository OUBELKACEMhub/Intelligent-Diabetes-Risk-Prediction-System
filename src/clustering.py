import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_and_preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def apply_clustering():
    df, df_scaled = load_and_preprocess_data()
    
    if "id" in df_scaled.columns:
        df_scaled_clean = df_scaled.drop(columns=["id"])
    else:
        df_scaled_clean = df_scaled

    # Entraînement K-Means (ex: k=3 choisi via la méthode du coude/silhouette)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled_clean)
    
    df['Cluster'] = clusters
    
    # Analyse des moyennes par cluster
    cluster_means = df.groupby('Cluster')[['Glucose', 'BMI', 'DiabetesPedigreeFunction']].mean()
    print("\nMoyennes par cluster :\n", cluster_means)
    
    # Identification du cluster à haut risque (Glucose > 126, BMI > 30, DPF > 0.5)
    high_risk_clusters = cluster_means[
        (cluster_means['Glucose'] > 126) &
        (cluster_means['BMI'] > 30) &
        (cluster_means['DiabetesPedigreeFunction'] > 0.5)
    ].index

    print("Clusters à haut risque identifiés :", list(high_risk_clusters))

    
    df['risk_category'] = df['Cluster'].apply(lambda x: 1 if x in high_risk_clusters else 0)
    
    # Sauvegarde du dataset enrichi avec les risque
    risk_path = DATA_DIR / "processed" / "pima_diabetes_clusters_risk.csv"
    df.to_csv(risk_path, index=False)
    print("Dataset avec risk_category sauvegardé.")
    
    return df
 
if __name__ == "__main__":
    apply_clustering()