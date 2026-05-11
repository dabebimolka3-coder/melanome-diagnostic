import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from datetime import datetime

# --- 1. FONCTIONS TECHNIQUES (À placer avant tout le reste) ---

def get_base64_of_bin_file(bin_file):
    """Encode l'image locale en base64 pour le CSS"""
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

@st.cache_resource
def load_assets():
    """Charge le modèle et les paramètres JSON"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_multimodal_54.pkl")
    json_path = os.path.join(current_dir, "params_multimodal_54.json")
    try:
        model = joblib.load(model_path)
        with open(json_path, 'r') as f:
            params = json.load(f)
        return model, params
    except Exception as e:
        st.error(f"Erreur de chargement des ressources : {e}")
        return None, None

# Initialisation des variables
img_name = "mountainviewrheumatoidarthritis.webp"
img_base64 = get_base64_of_bin_file(img_name)
model, params = load_assets()

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MelanomaPredict AI | Portail Diagnostic",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. STYLE CSS PERSONNALISÉ (AVEC IMAGE DE FOND) ---
if img_base64:
    bg_style = f"""
    .stApp {{
        background-image: url("data:image/webp;base64,{img_base64}");
        background-attachment: fixed;
        background-size: cover;
    }}
    """
else:
    bg_style = ".stApp { background-color: #f4f7f9; }"

st.markdown(f"""
    <style>
    {bg_style}
    
    /* Calque de lisibilité */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(244, 247, 249, 0.91); 
        z-index: -1;
    }}

    /* Sidebar sombre */
    [data-testid="stSidebar"] {{
        background-color: #0e1117 !important;
        color: white;
    }}

    /* Header principal */
    .main-header {{
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    /* Cartes blanches avec ombres */
    .report-card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef2f6;
        margin-bottom: 1.2rem;
    }}

    /* Bouton Diagnostic */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        height: 3rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. CONTENU PRINCIPAL ---

st.markdown("""
    <div class="main-header">
        <h1>MelanomaPredict AI 🧬</h1>
        <p>Analyse Multimodale pour l'aide à la décision thérapeutique du Mélanome</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie"])

# --- ONGLET MÉTHODOLOGIE ---
with tab2:
    st.header("📖 Méthodologie Scientifique")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Architecture")
        st.markdown("""
        * **Biomarqueurs :** 54 gènes (mRNA) sélectionnés par Lasso.
        * **Modèle :** Random Forest (500 arbres).
        * **Données :** Entraînement sur TCGA-SKCM.
        """)
    with col_b:
        st.subheader("🚀 Procédure Clinique")
        st.info("Saisie des données -> Normalisation Z-score -> Calcul de probabilité métastatique.")

# --- ONGLET ANALYSE ---
with tab1:
    col_input, col_display = st.columns([1, 1.6], gap="large")
    
    with col_input:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("📋 Paramètres Patient")
        age = st.number_input("Âge", 1, 115, 55)
        sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        stade = st.select_slider("Stade Initial", options=["I", "II", "III", "IV"])
        
        st.divider()
        st.subheader("📥 Données Omiques")
        if params:
            example_df = pd.DataFrame(np.random.uniform(0.5, 5.0, size=(1, 54)), columns=params['top_genes'])
            st.download_button("📥 Télécharger Template", example_df.to_csv(index=False).encode('utf-8'), "template.csv")
        
        uploaded_file = st.file_uploader("Charger le profil (.csv)", type="csv")
        
        if uploaded_file and model and params:
            if st.button("Lancer le Diagnostic"):
                with st.spinner("Calcul en cours..."):
                    df_p = pd.read_csv(uploaded_file)
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
                    # Fusion et Scaling
                    X_combined = np.array([age, sexe_val, stade_map[stade]] + df_p[params['top_genes']].iloc[0].tolist()).reshape(1, -1)
                    X_scaled = (X_combined - np.array(params['means'])) / np.array(params['stds'])
                    st.session_state['analysis'] = {'prob': model.predict_proba(X_scaled)[0][1], 'genes': params['top_genes']}
        st.markdown('</div>', unsafe_allow_html=True)

    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            p = res['prob']
            
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📊 Rapport d'Analyse")
            st.metric("Risque Métastatique", f"{p*100:.1f}%")
            st.progress(p)

            if p < 0.33: st.success("🟢 RISQUE FAIBLE")
            elif p < 0.67: st.warning("🟡 RISQUE INTERMÉDIAIRE")
            else: st.error("🔴 RISQUE ÉLEVÉ")
            st.markdown('</div>', unsafe_allow_html=True)

            # Importance des gènes
            importances = model.feature_importances_[3:]
            top_10 = pd.DataFrame({'Gène': res['genes'], 'Imp': importances}).sort_values('Imp', ascending=False).head(10)
            fig = px.bar(top_10, x='Imp', y='Gène', orientation='h', color='Imp', color_continuous_scale='Reds', title="Top 10 Biomarqueurs")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complétez le profil à gauche pour générer l'analyse.")

# BARRE LATÉRALE
with st.sidebar:
    st.title("⚙️ Système")
    if model: st.success("Modèle : OK")
    else: st.error("Erreur de chargement")
    st.caption(f"📅 Date : {datetime.now().strftime('%d/%m/%Y')}")
