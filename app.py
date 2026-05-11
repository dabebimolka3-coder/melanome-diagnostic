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
        # En cas d'erreur, affiche un message mais ne bloque pas l'exécution
        # (les boutons de diagnostic ne fonctionneront pas, mais l'interface s'affichera)
        return None, None

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MedFlow | MelanomaPredict AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CHARGEMENT ET ENCODAGE DE L'IMAGE DE FOND ---
img_name = "mountainviewrheumatoidarthritis.webp"
img_base64 = get_base64_of_bin_file(img_name)

# --- 4. CHARGEMENT DU MODÈLE ET DES PARAMÈTRES ---
model, params = load_assets()

# --- 5. STYLE CSS PERSONNALISÉ (DESIGN DASHBOARD & FOND MODERNE) ---

# Définir le style d'arrière-plan en fonction de la présence de l'image
if img_base64:
    bg_style = f"""
    .stApp {{
        background-image: url("data:image/webp;base64,{img_base64}");
        background-attachment: fixed;
        background-size: cover;
    }}
    """
else:
    # Fond gris clair par défaut si l'image est manquante
    bg_style = ".stApp { background-color: #f4f7f9; }"

# Injecter tout le CSS personnalisé
st.markdown(f"""
    <style>
    {bg_style}
    
    /* Calque de lisibilité semi-transparent sur toute l'application */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(244, 247, 249, 0.91); 
        z-index: -1;
    }}

    /* Sidebar (barre latérale) sombre */
    [data-testid="stSidebar"] {{
        background-color: #0e1117 !important;
        color: white;
    }}
    
    /* Couleur du texte dans la barre latérale */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #dfe3e6;
    }}

    /* En-tête principal (Main Header) */
    .main-header {{
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    /* Cartes blanches avec ombres douces pour formulaires et résultats */
    .report-card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef2f6;
        margin-bottom: 1.2rem;
    }}

    /* Style des onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px;
    }}

    /* Style du bouton Diagnostic (Lancer le Diagnostic) */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        height: 3rem;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background: linear-gradient(90deg, #001f40 0%, #003680 100%);
        color: white;
        border: none;
    }}
    
    /* Couleur du titre dans la barre latérale */
    .sidebar-title {{
        color: white !important;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 6. CONTENU DE L'APPLICATION (MAIN CONTENT) ---

# En-tête principal
st.markdown("""
    <div class="main-header">
        <h1>MedFlow | MelanomaPredict AI 🧬</h1>
        <p>Analyse Multimodale pour l'aide à la décision thérapeutique du Mélanome cutané</p>
    </div>
    """, unsafe_allow_html=True)

# Définition des onglets
tab1, tab2 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie"])

# --- CONTENU DE L'ONGLET MÉTHODOLOGIE ---
with tab2:
    st.header("📖 Méthodologie Scientifique & Technique")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Architecture du Modèle")
        st.markdown("""
        * **Signature Génomique :** Quantification de **54 biomarqueurs (mRNA)** sélectionnés par régression Lasso.
        * **Moteur de Prédiction :** Forêt Aléatoire (*Random Forest*) de 500 arbres décisionnels.
        * **Source :** Entraîné sur les cohortes transcriptomiques normalisées provenant du **TCGA-SKCM**.
        """)
    with col_b:
        st.subheader("🚀 Procédure Clinique pour le Clinicien")
        st.info("""
        **1. Input :** Saisie clinique (âge, sexe, stade) et chargement du fichier `.csv` (54 gènes HGNC).
        **2. Processing :** Fusion des données (57 variables) et standardisation **Z-score** ($$z = \\frac{x - \\mu}{\\sigma}$$).
        **3. Output :** Calcul de la probabilité $p$ métastatique et recommandation thérapeutique automatisée.
        """)

# --- CONTENU DE L'ONGLET ANALYSE PATIENT ---
with tab1:
    # Création de deux colonnes principales : Inputs et Affichage
    col_input, col_display = st.columns([1, 1.6], gap="large")
    
    # Colonne de gauche : Entrées (Configuration)
    with col_input:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("📋 Paramètres Patient")
        age = st.number_input("Âge du patient", 1, 115, 55)
        sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        stade = st.select_slider("Stade Initial AJCC", options=["I", "II", "III", "IV"])
        
        st.divider()
        st.subheader("📥 Données Omiques")
        
        # Vérification si les paramètres ont été chargés pour générer le template
        if params:
            # Générer un DataFrame d'exemple avec des valeurs aléatoires pour le template
            example_df = pd.DataFrame(np.random.uniform(0.5, 5.0, size=(1, 54)), columns=params['top_genes'])
            # Ajouter un bouton de téléchargement pour le template
            st.download_button(
                label="📥 Télécharger Template (.csv)",
                data=example_df.to_csv(index=False).encode('utf-8'),
                file_name="template_medflow_54g.csv",
                mime="text/csv"
            )
        
        # Champ de chargement du fichier omique
        uploaded_file = st.file_uploader("Charger le profil d'expression (CSV)", type="csv")
        
        # Logique de traitement après chargement et clic sur le bouton
        if uploaded_file and model and params:
            if st.button("Lancer le Diagnostic", use_container_width=True):
                with st.spinner("Analyse du profil en cours..."):
                    # Lire le fichier patient chargé
                    df_p = pd.read_csv(uploaded_file)
                    
                    # Convertir les entrées cliniques en valeurs numériques pour le modèle
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
                    
                    # Préparation des données d'entrée combinées
                    clinique_vec = [age, sexe_val, stade_map[stade]]
                    # S'assurer que seules les colonnes correspondant aux 54 gènes sont utilisées
                    omique_vec = df_p[params['top_genes']].iloc[0].tolist()
                    
                    # Fusionner les vecteurs clinique et omique (57 features au total)
                    X_combined = np.array(clinique_vec + omique_vec).reshape(1, -1)
                    
                    # Appliquer le scaling (Z-score) basé sur les moyennes et écarts-types de la cohorte TCGA
                    X_scaled = (X_combined - np.array(params['means'])) / np.array(params['stds'])
                    
                    # Calculer la probabilité métastatique avec le modèle
                    prob = model.predict_proba(X_scaled)[0][1]
                    
                    # Stocker les résultats dans la session state pour un affichage persistant
                    st.session_state['analysis'] = {
                        'prob': prob,
                        'genes': params['top_genes'],
                        'model_importances': model.feature_importances_
                    }
        st.markdown('</div>', unsafe_allow_html=True)

    # Colonne de droite : Affichage des résultats
    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            p = res['prob']
            
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("📊 Rapport d'Interprétation Diagnostic")
            
            # Affichage de la métrique principale et de la barre de progression
            st.metric("Score de Risque Métastatique (Probabilité p)", f"{p*100:.1f}%")
            st.progress(p)

            # Logique de décision clinique automatisée basée sur les seuils définis
            if p < 0.33:
                st.success("🟢 **RISQUE FAIBLE** — Mélanome primaire probable. Surveillance standard recommandée.")
            elif p < 0.67:
                st.warning("🟡 **RISQUE INTERMÉDIAIRE** — Zone d'incertitude. Suivi rapproché et examens complémentaires requis.")
            else:
                st.error("🔴 **RISQUE ÉLEVÉ** — Mélanome métastatique probable. Discussion précoce d'immunothérapie/thérapie ciblée.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Visualisation des 10 biomarqueurs les plus décisifs ---
            # Exclure les 3 premières importances (cliniques) pour se concentrer sur les gènes
            importances_genes = res['model_importances'][3:]
            
            # Créer un DataFrame pour le graphique Plotly
            top_10_df = pd.DataFrame({
                'Gène': res['genes'],
                'Importance': importances_genes
            }).sort_values('Importance', ascending=False).head(10)
            
            # Créer le graphique à barres horizontales avec Plotly Express
            fig = px.bar(
                top_10_df,
                x='Importance',
                y='Gène',
                orientation='h',
                color='Importance',
                color_continuous_scale='Reds',
                title="Top 10 des Biomarqueurs les plus Décisifs",
                template="simple_white"
            )
            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Veuillez configurer les paramètres du patient et charger un fichier omique valide (.csv) dans la colonne de gauche pour lancer l'analyse.")

# --- 7. BARRE LATÉRALE (SIDEBAR) AVEC STATUT SYSTÈME ET AIDE ---
with st.sidebar:
    st.markdown('<h3 class="sidebar-title">⚙️ Statut Système</h3>', unsafe_allow_html=True)
    if model:
        st.success("Modèle Multimodal : OK")
        st.caption("Version de l'algorithme : 2.0.2")
        st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y')}")
    else:
        st.error("Erreur de chargement des ressources (Modèle ou JSON)")
    
    st.divider()
    st.markdown('<h3 class="sidebar-title">🔍 Aide & Spécifications</h3>', unsafe_allow_html=True)
    st.markdown("""
    **Format du fichier Omique :**
    - Format : `.csv` (séparateur virgule `,`)
    - Colonnes : Utiliser les symboles officiels HGNC (ex: *MMP3*, *MARCO*, *LYVE1*).
    - Un template est disponible au téléchargement.
    
    **Interprétation :**
    - Le score est basé sur un modèle bio-informatique. Il ne remplace pas l'avis d'un oncologue et doit être corrélé aux examens cliniques.
    """)
