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
    page_title="MelanomaPredict AI | Portail Diagnostic",
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_multimodal_54.pkl")
    json_path = os.path.join(current_dir, "params_multimodal_54.json")
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
        <h1>MelanomaPredict AI 🧬</h1>
        <p>Analyse Multimodale pour l'aide à la décision thérapeutique du Mélanome cutané</p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. NAVIGATION ---
# Correction : 2 onglets correspondent à 2 variables
tab1, tab2 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie"])

# --- MÉTHODOLOGIE ---
with tab2:
    st.header("📖 Méthodologie Scientifique & Technique")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Architecture du Modèle")
        st.markdown("""
        * **Signature Génomique :** Quantification de **54 biomarqueurs (mRNA)** sélectionnés par régression Lasso. Ces gènes sont impliqués dans l'invasion, le Remodelage de la Matrice (MEC), l'EMT et l'Inflammation.
        * **Moteur de Prédiction :** Forêt Aléatoire (*Random Forest*) de 500 arbres décisionnels.
        * **Source :** Entraîné sur les cohortes **TCGA-SKCM**.
        """)
        
        st.subheader("2. Prétraitement & Normalisation")
        st.markdown(r"""
        Chaque échantillon subit une normalisation **Z-score** basée sur les paramètres de la cohorte de référence :
        $$z = \frac{x - \mu}{\sigma}$$
        """)

    with col_b:
        st.subheader("🚀 Procédure de Diagnostic pour le Clinicien")
        st.info("""
        **Étape 1 : Préparation (Input)**
        - Saisie manuelle des paramètres cliniques (Âge, Sexe, Stade).
        - Chargement du profil d'expression (54 gènes) au format `.csv`.

        **Étape 2 : Traitement (Processing)**
        - **Fusion Multimodale :** Encodage et concaténation pour former un profil unique de 57 variables.
        - **Standardisation :** Application des moyennes et écart-types du TCGA.

        **Étape 3 : Aide à la Décision (Output)**
        - Calcul de la probabilité $p$ métastatique.
        - Recommandation clinique automatisée selon le score obtenu.
        """)

# --- ANALYSE PATIENT ---
with tab1:
    st.warning("**Dispositif de Recherche** : Ce système génère un score de risque basé sur l'analyse de 54 signatures transcriptomiques.")
    
    col_input, col_display = st.columns([1, 2], gap="large")
    
    with col_input:
        st.subheader("📋 Paramètres Cliniques")
        age = st.number_input("Âge du patient", 1, 115, 55)
        sexe = st.radio("Sexe", ["Homme", "Femme"])
        stade = st.selectbox("Stade Initial", ["I", "II", "III", "IV"])
        
        st.divider()
        st.subheader("📥 Données omiques")
        
        if params:
            example_df = pd.DataFrame(np.random.uniform(0.5, 5.0, size=(1, 54)), columns=params['top_genes'])
            csv_example = example_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Télécharger le Template (.csv)",
                data=csv_example,
                file_name="template_54_genes.csv",
                mime="text/csv"
            )
        
        uploaded_file = st.file_uploader("Charger le profil d'expression (HGNC Symbols)", type="csv")
        
        if uploaded_file and model and params:
            df_patient = pd.read_csv(uploaded_file)
            if st.button("Lancer le Diagnostic", use_container_width=True):
                with st.spinner("Analyse du profil en cours..."):
                    # Encodage
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
                    stade_val = stade_map[stade]
                    
                    # Préparation des données
                    clinique_vec = [age, sexe_val, stade_val]
                    omique_vec = df_patient[params['top_genes']].iloc[0].tolist()
                    X_combined = np.array(clinique_vec + omique_vec).reshape(1, -1)
                    
                    # Scaling & Prédiction
                    X_scaled = (X_combined - np.array(params['means'])) / np.array(params['stds'])
                    prob = model.predict_proba(X_scaled)[0][1]
                    
                    st.session_state['analysis'] = {'prob': prob, 'top_genes': params['top_genes']}

    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            prob_percent = res['prob']
            
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📊 Rapport d'Interprétation")
            
            # Métrique et Barre
            st.metric("Probabilité Métastatique (Score p)", f"{prob_percent*100:.1f}%")
            st.progress(prob_percent)

            # Logique de décision clinique
            if prob_percent < 0.33:
                st.success("🟢 **RISQUE FAIBLE** — Mélanome primaire probable")
                st.markdown("**Décision :** Surveillance standard et suivi dermatologique classique.")
            elif 0.33 <= prob_percent < 0.67:
                st.warning("🟡 **RISQUE INTERMÉDIAIRE** — Zone d'incertitude")
                st.markdown("**Décision :** Examens complémentaires, confirmation histologique et suivi rapproché.")
            else:
                st.error("🔴 **RISQUE ÉLEVÉ** — Mélanome métastatique probable")
                st.markdown("**Décision :** Discussion précoce d'immunothérapie (anti-PD-1 / anti-CTLA-4) ou thérapie ciblée.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Importance des Biomarqueurs
            # Correction : Suppression du mot 'import' devant importances
            importances = model.feature_importances_[3:]
            top_10_df = pd.DataFrame({'Gène': res['top_genes'], 'Imp': importances}).sort_values('Imp', ascending=False).head(10)
            
            fig = px.bar(top_10_df, x='Imp', y='Gène', orientation='h', color='Imp',
                         title="Top 10 des Biomarqueurs Décisifs",
                         color_continuous_scale='Reds', template="simple_white")
            st.plotly_chart(fig, use_container_width=True)

# BARRE LATÉRALE
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062231.png", width=80)
    st.title("Statut Système")
    if model:
        st.success("Modèle : Opérationnel")
        st.info("Algorithme : Random Forest")
        st.info("Features : 54 Gènes + 3 Cliniques")
    else:
        st.error("Ressources manquantes")
