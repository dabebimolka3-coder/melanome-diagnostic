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

# --- 2. STYLE CSS PERSONNALISÉ (DESIGN DASHBOARD) ---
st.markdown("""
    <style>
    /* Couleur de fond de l'application */
    .stApp {
        background-color: #f4f7f9;
    }
    
    /* Style de la barre latérale (Sidebar) sombre */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        color: white;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #dfe3e6;
    }

    /* En-tête principal */
    .main-header {
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Cartes de résultats et de formulaires */
    .report-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e6e9ef;
        margin-bottom: 1rem;
    }

    /* Style des onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* Bouton principal */
    .stButton>button {
        width: 100%;
        background-color: #004aad;
        color: white;
        border-radius: 6px;
        border: none;
        height: 3rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #003073;
        border: none;
        color: white;
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
tab1, tab2 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie"])

# --- MÉTHODOLOGIE ---
with tab2:
    st.header("📖 Méthodologie Scientifique & Technique")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. Architecture du Modèle")
        st.markdown("""
        * **Signature Génomique :** Quantification de **54 biomarqueurs (mRNA)** sélectionnés par régression Lasso.
        * **Moteur de Prédiction :** Forêt Aléatoire (*Random Forest*) de 500 arbres décisionnels.
        * **Source :** Entraîné sur les cohortes **TCGA-SKCM**.
        """)
        st.subheader("2. Prétraitement")
        st.markdown(r"Normalisation **Z-score** : $$z = \frac{x - \mu}{\sigma}$$")

    with col_b:
        st.subheader("🚀 Procédure de Diagnostic")
        st.info("""
        **1. Input :** Saisie clinique et chargement du fichier `.csv` (54 gènes).
        **2. Processing :** Fusion multimodale (57 variables) et standardisation.
        **3. Output :** Calcul du score $p$ et recommandation thérapeutique.
        """)

# --- ANALYSE PATIENT ---
with tab1:
    col_input, col_display = st.columns([1, 1.6], gap="large")
    
    with col_input:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("🧬 Configuration")
        age = st.number_input("Âge du patient", 1, 115, 55)
        sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        stade = st.select_slider("Stade Initial", options=["I", "II", "III", "IV"])
        
        st.divider()
        st.subheader("📥 Données omiques")
        if params:
            example_df = pd.DataFrame(np.random.uniform(0.5, 5.0, size=(1, 54)), columns=params['top_genes'])
            st.download_button("📥 Template (.csv)", example_df.to_csv(index=False).encode('utf-8'), "template.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Fichier d'expression", type="csv")
        
        if uploaded_file and model and params:
            if st.button("Lancer le Diagnostic"):
                with st.spinner("Analyse en cours..."):
                    df_patient = pd.read_csv(uploaded_file)
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
                    X_combined = np.array([age, sexe_val, stade_map[stade]] + df_patient[params['top_genes']].iloc[0].tolist()).reshape(1, -1)
                    X_scaled = (X_combined - np.array(params['means'])) / np.array(params['stds'])
                    st.session_state['analysis'] = {'prob': model.predict_proba(X_scaled)[0][1], 'top_genes': params['top_genes']}
        st.markdown('</div>', unsafe_allow_html=True)

    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            p = res['prob']
            
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📊 Rapport d'Interprétation")
            st.metric("Score de Risque Métastatique", f"{p*100:.1f}%")
            st.progress(p)

            if p < 0.33:
                st.success("🟢 **RISQUE FAIBLE** — Surveillance standard recommandée.")
            elif p < 0.67:
                st.warning("🟡 **RISQUE INTERMÉDIAIRE** — Suivi rapproché requis.")
            else:
                st.error("🔴 **RISQUE ÉLEVÉ** — Discussion thérapeutique (Immunothérapie).")
            st.markdown('</div>', unsafe_allow_html=True)

            # Importance des Biomarqueurs
            importances = model.feature_importances_[3:]
            top_10 = pd.DataFrame({'Gène': res['top_genes'], 'Imp': importances}).sort_values('Imp', ascending=False).head(10)
            fig = px.bar(top_10, x='Imp', y='Gène', orientation='h', color='Imp', color_continuous_scale='Reds', template="simple_white", title="Top 10 Biomarqueurs Décisifs")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veuillez configurer les paramètres et charger un fichier pour lancer l'analyse.")

# BARRE LATÉRALE
with st.sidebar:
    st.markdown("### 🛠️ Statut Système")
    if model:
        st.success("Modèle Multimodal : OK")
        st.caption("Version de l'algorithme : 2.0.1")
        st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y')}")
    else:
        st.error("Erreur de chargement")
    st.divider()
    st.markdown("🔍 **Aide** : Assurez-vous que le fichier CSV utilise les symboles officiels HGNC.")
