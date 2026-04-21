import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION INITIALE ---
st.set_page_config(
    page_title="Plateforme Officielle",
    page_icon="🧬",
    layout="wide"
)

# --- THEME & DESIGN CSS ---
st.markdown("""
    <style>
    /* Style général */
    .main { background-color: #f0f2f6; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #004aad 0%, #002d6b 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Cartes de résultats */
    .metric-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-top: 5px solid #004aad;
    }
    
    /* Boutons */
    .stButton>button {
        background-color: #004aad;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, "model_final_pure.pkl"))
        with open(os.path.join(base_path, "params_pures.json"), 'r') as f:
            params = json.load(f)
        return model, params
    except:
        return None, None

rf_model, scaling_params = load_assets()

# --- SIGNATURE GÉNÉTIQUE ---
GENES = ['CLDN11', 'GSC2', 'IL11', 'MMP9', 'HYAL4', 'ASIC2', 'NMU', 'TSPAN11', 'IL17F', 'FGF12', 'CHST8', 'PHAF1', 'HOXD11', 'LBP', 'GFAP', 'LIN28A', 'VPS33A', 'ZNF385B', 'SLC5A12', 'GRM1', 'CNTNAP4', 'LGI2', 'KCNJ16', 'CHODL', 'SEPTIN14', 'ADAMTSL3', 'GDF6', 'ZIC3', 'ITLN2', 'ALOX15', 'XIRP2', 'IGFN1', 'CCDC141', 'CDS1', 'SLC30A8', 'PSCA', 'CST2', 'KRT4', 'OTP', 'CALB2', 'CNTNAP2', 'CCDC197', 'LRRN1', 'KLHL38', 'CPN2', 'WSCD1', 'ABCA13', 'C1orf105', 'MAPK15', 'FDCSP', 'KRTAP13-2', 'TFDP3', 'CFAP91', 'ADARB2', 'MYT1L', 'HRCT1', 'SLC30A10', 'TLE1', 'COL4A6', 'PNLIPRP3', 'FER1L6', 'STMND1', 'NOX5']

# --- INTERFACE : HEADER ---
st.markdown("""
    <div class="main-header">
        <h1> Prédiction du Mélanome cutané par IA </h1>
        <p>Système Expert d'Analyse Génomique pour le Diagnostic du Mélanome</p>
    </div>
    """, unsafe_allow_html=True)

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["🚀 Diagnostic", "📚 Documentation", "⚙️ Paramètres"])

with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.subheader("📥 Données Patient")
        uploaded_file = st.file_uploader("Téléverser le profil d'expression (CSV)", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Fichier chargé.")
            
            if st.button("Lancer l'analyse"):
                with st.spinner("Analyse des biomarqueurs en cours..."):
                    # Logique de prédiction
                    data = df[GENES]
                    means, stds = np.array(scaling_params['means']), np.array(scaling_params['stds'])
                    data_scaled = (data - means) / stds
                    
                    prob_malignite = rf_model.predict_proba(data_scaled)[0][1]
                    confiance = max(rf_model.predict_proba(data_scaled)[0]) * 100
                    
                    # Stockage en session
                    st.session_state['results'] = {
                        'prob': prob_malignite,
                        'conf': confiance,
                        'data_scaled': data_scaled
                    }

    with col_right:
        if 'results' in st.session_state:
            res = st.session_state['results']
            
            st.subheader("📊 Résultats du Diagnostic")
            
            # Affichage Verdict
            if res['prob'] > 0.5:
                st.error(f"### CLASSIFICATION : MÉLANOME (Maligne)")
            else:
                st.success(f"### CLASSIFICATION : BÉNIN (Sain)")
            
            c1, c2 = st.columns(2)
            c1.metric("Probabilité Maligne", f"{res['prob']*100:.1f}%")
            c2.metric("Indice de Confiance", f"{res['conf']:.2f}%")
            
            st.progress(float(res['prob']))
            
            # Graphique
            st.markdown("---")
            st.write("**Top 5 Biomarqueurs Décisifs**")
            importances = rf_model.feature_importances_
            contributions = (res['data_scaled'].values[0] * importances)
            top_5 = pd.DataFrame({'Gène': GENES, 'C': contributions}).sort_values('C', key=abs, ascending=False).head(5)
            
            fig = px.bar(top_5, x='C', y='Gène', orientation='h', color='C',
                         color_continuous_scale='RdYlGn_r', template="simple_white")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Notes Techniques")
    st.write("""
    Cette plateforme utilise un modèle de **Random Forest** entraîné sur des profils d'expression génique.
    - **Nombre de gènes** : 63 biomarqueurs sélectionnés par importance statistique.
    - **Normalisation** : Z-score calculé sur la cohorte d'entraînement.
    - **Usage** : Aide au diagnostic uniquement.
    """)
    

with tab3:
    st.subheader("Configuration Système")
    st.write(f"Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d')}")
    if rf_model:
        st.write("✅ Modèle chargé : `model_final_pure.pkl`")
    if scaling_params:
        st.write("✅ Paramètres de scaling : `params_pures.json` (Format sécurisé)")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Tous droits réservés. Usage strictement confidentiel.")