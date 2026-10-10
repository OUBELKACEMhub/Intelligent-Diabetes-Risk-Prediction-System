import streamlit as st
import pandas as pd
import requests

# Configuration de la page Streamlit
st.set_page_config(page_title="Medical Risk Prediction - Praticien", page_icon="🩺", layout="centered")

st.title("🩺 Interface Praticien : Prédiction du Risque de Diabète")
st.markdown("Veuillez saisir les constantes cliniques du patient pour évaluer son niveau de risque via l'API de Production (MLflow).")

# URL de l'API FastAPI
API_URL = "http://localhost:8000/predict"

# Formulaire interactif pour la saisie des constantes cliniques
with st.form("clinical_form"):
    st.subheader("📋 Constantes du Patient")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Nombre de Grossesses (Pregnancies)", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Tression artérielle (Blood Pressure)", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Épaisseur de la peau (Skin Thickness)", min_value=0.0, max_value=100.0, value=20.0)
        
    with col2:
        insulin = st.number_input("Insuline", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("Indice de Masse Corporelle (BMI)", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Fonction Pédigrée du Diabète (DPF)", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Âge", min_value=1.0, max_value=120.0, value=30.0)

    # Bouton de soumission du formulaire
    submit_button = st.form_submit_button(label="🔍 Lancer l'Évaluation du Risque")

# Logique de prédiction après soumission (consommation de l'API FastAPI)
if submit_button:
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    
    try:
        # Appel de l'API FastAPI via une requête POST
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")
            risk_category = result.get("risk_category")
            probability = result.get("probability")
            
            st.markdown("---")
            st.subheader("📊 Résultat de l'Analyse (Via MLflow Production Model)")
            
            # Affichage avec indicateur visuel (Badge rouge / vert) et conseils de suivi
            if prediction == 1:
                st.error(f"⚠️ **Résultat : Patient à Risque Élevé (High Risk)**")
                if probability is not None:
                    st.write(f"Probabilité estimée : **{probability * 100:.2f}%**")
                
                # Conseils de suivi pour risque élevé
                st.warning("""
                **📋 Conseils de Suivi Médical (Risque Élevé) :**
                * Consultation médicale approfondie recommandée (bilan glycémique complet / HbA1c).
                * Adoption d'un régime alimentaire pauvre en sucres raffinés et graisses saturées.
                * Mise en place d'une activité physique régulière (marche, cardio).
                * Surveillance de la tension artérielle et du poids corporel.
                """)
            else:
                st.success(f"✅ **Résultat : Patient à Faible Risque (Low Risk)**")
                if probability is not None:
                    st.write(f"Probabilité estimée : **{probability * 100:.2f}%**")
                
                # Conseils de suivi pour risque faible
                st.info("""
                **📋 Conseils de Prévention (Risque Faible) :**
                * Maintenir un mode de vie sain et équilibré.
                * Continuer la pratique d'activités physiques régulières.
                * Contrôle de routine annuel recommandé.
                """)
        else:
            
            error_detail = response.json().get("detail", "Erreur inconnue")
            st.error(f"Erreur de l'API : {error_detail}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API FastAPI. Assurez-vous qu'elle est bien lancée sur `http://localhost:8000`.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")