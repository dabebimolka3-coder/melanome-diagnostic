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
    page_title="MelanomaPredict AI | Portail Officiel",
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
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .report-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_final_pure.pkl")
    json_path = os.path.join(current_dir, "params_pures.json")
    try:
        model = joblib.load(model_path)
        with open(json_path, 'r') as f:
            params = json.load(f)
        return model, params
    except:
        return None, None

rf_model, scaling_params = load_assets()

# SIGNATURE DES 63 GÈNES
GENES = ['CLDN11', 'GSC2', 'IL11', 'MMP9', 'HYAL4', 'ASIC2', 'NMU', 'TSPAN11', 'IL17F', 'FGF12', 'CHST8', 'PHAF1', 'HOXD11', 'LBP', 'GFAP', 'LIN28A', 'VPS33A', 'ZNF385B', 'SLC5A12', 'GRM1', 'CNTNAP4', 'LGI2', 'KCNJ16', 'CHODL', 'SEPTIN14', 'ADAMTSL3', 'GDF6', 'ZIC3', 'ITLN2', 'ALOX15', 'XIRP2', 'IGFN1', 'CCDC141', 'CDS1', 'SLC30A8', 'PSCA', 'CST2', 'KRT4', 'OTP', 'CALB2', 'CNTNAP2', 'CCDC197', 'LRRN1', 'KLHL38', 'CPN2', 'WSCD1', 'ABCA13', 'C1orf105', 'MAPK15', 'FDCSP', 'KRTAP13-2', 'TFDP3', 'CFAP91', 'ADARB2', 'MYT1L', 'HRCT1', 'SLC30A10', 'TLE1', 'COL4A6', 'PNLIPRP3', 'FER1L6', 'STMND1', 'NOX5']

# --- 4. HEADER PRINCIPAL ---
st.markdown("""
    <div class="main-header">
        <h1>MelanomaPredict AI v2.0</h1>
        <p>Système de Diagnostic Moléculaire Assisté par Intelligence Artificielle</p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie", "🤝 Collaboration", "⚖️ Mentions"])

with tab1:
    st.warning("""
    **AVERTISSEMENT LÉGAL ET SCIENTIFIQUE** : Ce système est un dispositif expérimental d'aide au diagnostic. 
    Il ne remplace en aucun cas l'examen clinique d'un oncologue. 
    Les résultats fournis sont des probabilités statistiques issues de l'analyse transcriptomique. 
    Usage strictement réservé à la recherche académique (Research Use Only).
    """)
    
    col_input, col_display = st.columns([1, 2], gap="large")
    
    with col_input:
        st.subheader("📥 Données Génomiques")
        uploaded_file = st.file_uploader("Charger le profil d'expression (CSV)", type="csv")
        
        if uploaded_file and rf_model:
            df_patient = pd.read_csv(uploaded_file)
            if st.button("Lancer le Protocole d'Analyse"):
                with st.spinner("Analyse moléculaire en cours..."):
                    data = df_patient[GENES]
                    means = np.array(scaling_params['means'])
                    stds = np.array(scaling_params['stds'])
                    data_scaled = (data - means) / stds
                    
                    prob = rf_model.predict_proba(data_scaled)[0][1]
                    conf = max(rf_model.predict_proba(data_scaled)[0]) * 100
                    st.session_state['analysis'] = {'prob': prob, 'conf': conf, 'scaled': data_scaled}

    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("Rapport Synthétique de Diagnostic")
            
            c1, c2 = st.columns(2)
            with c1:
                if res['prob'] > 0.5:
                    st.error("### VERDICT : MÉLANOME DÉTECTÉ")
                else:
                    st.success("### VERDICT : PROFIL BÉNIN")
            with c2:
                st.metric("Indice de Confiance", f"{res['conf']:.2f}%")
            
            st.write(f"**Probabilité de malignité :** {res['prob']*100:.1f}%")
            st.progress(float(res['prob']))
            st.markdown('</div>', unsafe_allow_html=True)

            # Visualisation des Biomarqueurs
            importances = rf_model.feature_importances_
            contributions = (res['scaled'].values[0] * importances)
            top_5 = pd.DataFrame({'Gène': GENES, 'C': contributions}).sort_values('C', key=abs, ascending=False).head(5)
            
            fig = px.bar(
                top_5, 
                x='C', 
                y='Gène', 
                orientation='h', 
                color='C', 
                title="Top 5 des Biomarqueurs Décisifs",
                color_continuous_scale='RdYlGn_r', 
                template="simple_white"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🔬 Architecture Scientifique")
    st.markdown("""
    #### 1. Signature Génomique
    Analyse de **63 gènes cibles** sélectionnés pour leur rôle clé dans l'oncogenèse cutanée.
    #### 2. Algorithme de Classification
    Modèle de Forêt Aléatoire (Random Forest) entraîné sur les cohortes du **The Cancer Genome Atlas (TCGA)**.
    #### 3. Normalisation des Données
    Application du Z-score pour assurer la comparabilité inter-échantillons.
    """)

with tab3:
    st.subheader("🤝 Collaboration et Support")
    st.write("**Contact Scientifique :** research@melanomapredict-ai.org")
    with st.form("contact"):
        st.text_input("Institution / Laboratoire")
        st.text_area("Objet de la collaboration")
        st.form_submit_button("Envoyer la demande")

with tab4:
    st.caption(f"Système Opérationnel - Mis à jour le : {datetime.now().strftime('%d/%m/%Y')}")
    st.caption("© 2026 MelanomaPredict AI | Technologie de Diagnostic Moléculaire")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062231.png", width=80)
    st.title("Statut du Système")
    if rf_model:
        st.success("Modèle : Opérationnel")
    else:
        st.error("Modèle : Erreur de chargement")
    st.info("Version Stable : 2.0.4")
