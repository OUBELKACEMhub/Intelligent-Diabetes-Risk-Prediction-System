from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def load_and_preprocess_data():
    print("--- Étape 1 & 2 : Chargement et Prétraitement ---")
    
    # Chargement des données brutes
    raw_path = DATA_DIR / "raw" / "dataset-diabete-6ab953aea55fc643490108.csv"
    df = pd.read_csv(raw_path)
    
    # Aperçu et vérification de base
    print("Dimensions du dataset :", df.shape)
    print("Valeurs manquantes :\n", df.isnull().sum())
    
    # Sauvegarde du dataset nettoyé (si ce n'est pas déjà fait)
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_path = processed_dir / "pima_diabetes_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    
    # Normalisation / Standardisation des variables numériques (hors id si présent)
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if "id" in numeric_cols:
        numeric_cols = numeric_cols.drop("id")
        
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    scaled_path = processed_dir / "data_Scaled.csv"
    df_scaled.to_csv(scaled_path, index=False)
    
    print("Prétraitement terminé et fichiers sauvegardés.")
    return df, df_scaled

if __name__ == "__main__":
    load_and_preprocess_data()