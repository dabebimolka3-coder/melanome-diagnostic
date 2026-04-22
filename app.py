import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MelanomaPredict AI | Diagnostic Métastatique",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    :root {
        --primary: #002b5c;
        --secondary: #f8f9fa;
        --accent: #004aad;
    }
    .main { background-color: var(--secondary); }
    .main-header {
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .report-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        border-left: 10px solid var(--primary);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    # Correction du nom de fichier si nécessaire (vérifiez params_multimodal_54 ou 63 sur GitHub)
    model_path = "model_multimodal_54.pkl"
    json_path = "params_multimodal_54.json" 
    try:
        model = joblib.load(model_path)
        with open(json_path, 'r') as f:
            params = json.load(f)
        return model, params
    except:
        return None, None

model, params = load_assets()

# --- 4. HEADER PRINCIPAL ---
st.markdown("""
    <div class="main-header">
        <h1>MelanomaPredict AI v2.1</h1>
        <p>Système de Diagnostic Multimodal (Omique & Clinique) du mélanome cutané </p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie LASSO", "🤝 Collaboration"])

with tab1:
    st.warning("**Usage Recherche Uniquement** : Classification binaire (Métastatique vs Non-Métastatique).")
    
    col_input, col_display = st.columns([1, 2], gap="large")
    
    with col_input:
        st.subheader("📋 Paramètres Cliniques")
        age = st.number_input("Âge du patient", 1, 115, 55)
        sexe = st.radio("Sexe", ["Homme", "Femme"])
        stade = st.selectbox("Stade Initial", ["I", "II", "III", "IV"])
        
        st.divider()
        st.subheader("📥 Données Omiques")
        uploaded_file = st.file_uploader("Charger le profil d'expression (CSV des 54 gènes)", type="csv")
        
        if uploaded_file and model and params:
            df_patient = pd.read_csv(uploaded_file)
            if st.button("Lancer le Diagnostic Multimodal"):
                with st.spinner("Fusion des données en cours..."):
                    # 1. Encodage clinique
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
                    stade_val = stade_map[stade]
                    
                    # 2. Préparation du vecteur
                    clinique_vec = [age, sexe_val, stade_val]
                    omique_vec = df_patient[params['top_genes']].iloc[0].tolist()
                    X_combined = np.array(clinique_vec + omique_vec).reshape(1, -1)
                    
                    # 3. Normalisation et Prédiction
                    X_scaled = (X_combined - np.array(params['means'])) / np.array(params['stds'])
                    prob = model.predict_proba(X_scaled)[0][1]
                    
                    st.session_state['analysis'] = {'prob': prob, 'top_genes': params['top_genes']}

    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            prob_percent = res['prob'] * 100
            
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("Rapport d'Analyse Métastatique")
            
            if prob_percent < 35:
                st.success("### VERDICT : PROFIL NON-MÉTASTATIQUE")
                st.write("Le profil multimodal suggère une faible probabilité de dissémination.")
            elif 35 <= prob_percent <= 65:
                st.warning("### VERDICT : RISQUE INTERMÉDIAIRE")
                st.write("⚠️ **Attention :** Profil suspect nécessitant des examens d'imagerie complémentaires.")
            else:
                st.error("### VERDICT : RISQUE MÉTASTATIQUE ÉLEVÉ")
                st.write("Forte signature moléculaire associée aux formes métastatiques.")

            # Affichage des métriques corrigé (SANS res['conf'])
            st.metric("Score de Risque Métastatique", f"{prob_percent:.1f}%")
            st.progress(res['prob'])
            st.caption("Fiabilité du modèle : ~92% (basé sur les scores de validation croisée)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Importance des gènes
            importances = model.feature_importances_[3:]
            top_10_df = pd.DataFrame({'Gène': res['top_genes'], 'Imp': importances}).sort_values('Imp', ascending=False).head(10)
            
            fig = px.bar(top_10_df, x='Imp', y='Gène', orientation='h', color='Imp',
                         title="Top 10 des Biomarqueurs LASSO Décisifs",
                         color_continuous_scale='Reds', template="simple_white")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🔬 Rigueur Scientifique")
    st.markdown("Signature de **54 gènes** identifiés par LASSO sur les données TCGA.")

with tab3:
    st.write("© 2026 MelanomaPredict AI | research@melanomapredict-ai.org")

# BARRE LATÉRALE
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062231.png", width=80)
    st.title("Statut du Système")
    if model:
        st.success("Modèle Multimodal : Opérationnel")
    else:
        st.error("Erreur : Fichiers manquants")
