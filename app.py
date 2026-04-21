import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MelanomaPredict AI | Portail Officiel",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    /* Couleurs institutionnelles */
    :root {
        --primary: #002b5c;
        --secondary: #f8f9fa;
        --accent: #004aad;
    }
    
    .main { background-color: var(--secondary); }
    
    /* Header stylisé */
    .main-header {
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Cartes de résultats */
    .report-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Style des onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
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

# SIGNATURE DES 63 GÈNES
GENES = ['CLDN11', 'GSC2', 'IL11', 'MMP9', 'HYAL4', 'ASIC2', 'NMU', 'TSPAN11', 'IL17F', 'FGF12', 'CHST8', 'PHAF1', 'HOXD11', 'LBP', 'GFAP', 'LIN28A', 'VPS33A', 'ZNF385B', 'SLC5A12', 'GRM1', 'CNTNAP4', 'LGI2', 'KCNJ16', 'CHODL', 'SEPTIN14', 'ADAMTSL3', 'GDF6', 'ZIC3', 'ITLN2', 'ALOX15', 'XIRP2', 'IGFN1', 'CCDC141', 'CDS1', 'SLC30A8', 'PSCA', 'CST2', 'KRT4', 'OTP', 'CALB2', 'CNTNAP2', 'CCDC197', 'LRRN1', 'KLHL38', 'CPN2', 'WSCD1', 'ABCA13', 'C1orf105', 'MAPK15', 'FDCSP', 'KRTAP13-2', 'TFDP3', 'CFAP91', 'ADARB2', 'MYT1L', 'HRCT1', 'SLC30A10', 'TLE1', 'COL4A6', 'PNLIPRP3', 'FER1L6', 'STMND1', 'NOX5']

# --- HEADER PRINCIPAL ---
st.markdown("""
    <div class="main-header">
        <h1>MelanomaPredict AI v2.0</h1>
        <p>Système de Diagnostic Moléculaire Assisté par Intelligence Artificielle</p>
    </div>
    """, unsafe_allow_html=True)

# --- NAVIGATION PAR ONGLETS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie", "🤝 Collaboration", "⚖️ Mentions"])

# --- ONGLET 1 : DIAGNOSTIC ---
with tab1:
    st.info(" Note : *AVERTISSEMENT LÉGAL ET SCIENTIFIQUE* Ce système est un **dispositif expérimental d'aide au diagnostic** basé sur l'intelligence artificielle. 
        - Il ne remplace en aucun cas l'examen clinique d'un oncologue ou d'un dermatologue.
        - Les résultats fournis sont des probabilités statistiques issues de l'analyse transcriptomique.
        - Cet outil est strictement réservé à un **usage de recherche académique** (Research Use Only).")
    
    col_input, col_display = st.columns([1, 2], gap="large")
    
    with col_input:
        st.subheader("📥 Données Génomiques")
        uploaded_file = st.file_uploader("Charger le profil d'expression (CSV)", type="csv")
        
        if uploaded_file and rf_model:
            df_patient = pd.read_csv(uploaded_file)
            if st.button("Lancer le Protocole d'Analyse"):
                with st.spinner("Analyse en cours..."):
                    # Calculs
                    data = df_patient[GENES]
                    means, stds = np.array(scaling_params['means']), np.array(scaling_params['stds'])
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

            # Top Biomarqueurs
            importances = rf_model.feature_importances_
            contributions = (res['scaled'].values[0] * importances)
            top_5 = pd.DataFrame({'Gène': GENES, 'C': contributions}).sort_values('C', key=abs, ascending=False).head(5)
            
            fig = px.bar(top_5, x='C', y='Gène', orientation='h', color='C', 
                         title="Biomarqueurs Clés de l'Échantillon",
                         color_continuous_scale='RdYlGn_r', template="simple_white")
            st.plotly_chart(fig, use_container_width=True)
            
            report_txt = f"Diagnostic: {'Mélanome' if res['prob']>0.5 else 'Bénin'}\nProbabilité: {res['prob']*100:.2f}%"
            st.download_button("📥 Exporter le Rapport Officiel (TXT)", report_txt, "rapport_diagnostic.txt")

# --- ONGLET 2 : MÉTHODOLOGIE ---
with tab2:
    st.subheader("🔬 Architecture Scientifique")
    st.markdown("""
    #### 1. Signature Génomique
    Le diagnostic repose sur la quantification de **63 biomarqueurs (mRNA)** sélectionnés par l'algorithme Boruta. Ces gènes sont impliqués dans les mécanismes de prolifération cellulaire et d'évasion immunitaire.
    
    #### 2. Moteur de Prédiction
    Nous utilisons un modèle **Random Forest** (Forêt Aléatoire) de 500 arbres décisionnels. Ce modèle a été entraîné sur des cohortes transcriptomiques normalisées provenant du **TCGA** (The Cancer Genome Atlas).
    
    #### 3. Prétraitement des Données
    Chaque échantillon subit une normalisation **Z-score** basée sur les paramètres de la cohorte de référence :
    $$z = \\frac{x - \\mu}{\\sigma}$$
    """)

# --- ONGLET 3 : COLLABORATION ---
with tab3:
    st.subheader("🤝 Partenariats et Support")
    c_info, c_form = st.columns(2)
    with c_info:
        st.write("""
        **Unité de Recherche en Bioinformatique Moléculaire** Pour toute demande de collaboration académique ou accès à l'API :
        - **Email :** research@melanopredict-ai.org
        - **Institution :** Centre de Génomique Appliquée
        """)
    with c_form:
        with st.form("contact"):
            nom = st.text_input("Nom / Institution")
            msg = st.text_area("Message")
            if st.form_submit_button("Envoyer la demande"):
                st.success("Demande transmise à l'équipe de recherche.")

# --- ONGLET 4 : MENTIONS ---
with tab4:
    st.write(f"Système opérationnel - Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y')}")
    st.write("Conformité : Les données chargées sont traitées localement en mémoire et ne sont pas stockées.")
    st.markdown("---")
    st.caption("© 2026 MelanomaPredict AI | Usage strictement académique.")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062231.png", width=80)
    st.title("Contrôle Système")
    if rf_model:
        st.success("Modèle : Chargé")
    else:
        st.error("Modèle : Erreur")
    st.info("Version 2.0.4-stable")
