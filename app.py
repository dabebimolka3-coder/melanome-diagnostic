import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MelanomaPredict AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE POUR LA NAVIGATION ─────────────────────────────────────────
# Initialisation avec query params
if 'current_page' not in st.session_state:
    # Lire depuis query params ou défaut
    page_param = st.query_params.get("page", "analyse")
    st.session_state['current_page'] = page_param

# Fonction de navigation
def navigate_to(page):
    st.session_state['current_page'] = page
    st.query_params["page"] = page
    st.rerun()

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Syne:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff !important;
    color: #1a1a2e;
}
[data-testid="stAppViewContainer"] > .main { background: #ffffff !important; }
[data-testid="block-container"] { padding: 0 2.5rem 3rem !important; max-width: 1400px; }
* { font-family: 'Syne', sans-serif; }
h1, h2, h3, h4, h5, h6 { font-family: 'Cormorant Garamond', serif !important; color: #0d1b2a !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.topbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(0,0,0,0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 62px;
    margin: 0 -2.5rem 0;
}
.topbar-brand { display: flex; align-items: center; gap: 10px; }
.topbar-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1a6fff, #00c9a7);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: white;
}
.topbar-name {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0d1b2a;
    letter-spacing: 0.03em;
}
.nav-container {
    display: flex;
    gap: 1.5rem;
    align-items: center;
}
.nav-link {
    font-size: 0.7rem;
    color: rgba(0,0,0,0.45);
    text-decoration: none;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    cursor: pointer;
    transition: color 0.2s;
    background: none;
    border: none;
    font-family: 'Syne', sans-serif;
}
.nav-link:hover { color: #1a6fff; }
.nav-link.active { color: #1a6fff; }
.nav-separator {
    color: rgba(0,0,0,0.15);
    font-size: 0.7rem;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.72rem;
    color: rgba(0,0,0,0.45);
    letter-spacing: 0.06em;
}
.pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00c9a7;
    box-shadow: 0 0 0 0 rgba(0,201,167,0.6);
    animation: pulse 2s infinite;
    display: inline-block;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,201,167,0.6); }
    70%  { box-shadow: 0 0 0 8px rgba(0,201,167,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,201,167,0); }
}

.hero {
    position: relative; min-height: 340px;
    margin: 0 -2.5rem 3rem;
    overflow: hidden; display: flex; align-items: flex-end;
}
.hero-img {
    position: absolute; inset: 0;
    background-image: url('https://images.squarespace-cdn.com/content/v1/5d9e30182db9d71681f4a692/1581717140307-89XZXBK2C5OBW2AGDXCO/mountainviewrheumatoidarthritis.jpg');
    background-size: cover; background-position: center 25%;
    filter: brightness(0.9) saturate(1.1);
}
.hero-gradient {
    position: absolute; inset: 0;
    background: linear-gradient(to top, #ffffff 0%, rgba(255,255,255,0.85) 55%, rgba(255,255,255,0.6) 100%);
}
.hero-content {
    position: relative; z-index: 2;
    padding: 0 3rem 2.8rem; max-width: 780px;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(26,111,255,0.1);
    border: 1px solid rgba(26,111,255,0.25);
    border-radius: 50px; padding: 5px 14px;
    font-size: 0.68rem; color: #1a6fff;
    letter-spacing: 0.12em; text-transform: uppercase;
    font-weight: 600; margin-bottom: 1.1rem;
}
.hero-eyebrow-dot {
    width: 5px; height: 5px; border-radius: 50%; background: #1a6fff;
    display: inline-block;
}
.hero h1 {
    font-size: 3.2rem !important; font-weight: 600 !important;
    color: #0d1b2a !important; line-height: 1.1 !important;
    margin: 0 0 1rem !important; letter-spacing: -0.01em;
}
.hero h1 span { color: #1a6fff; }
.hero-sub {
    color: rgba(0,0,0,0.55); font-size: 0.9rem;
    line-height: 1.7; max-width: 520px;
}
.hero-kpis {
    display: flex; gap: 0; margin-top: 2rem;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px; overflow: hidden; width: fit-content;
    background: rgba(255,255,255,0.6);
}
.kpi {
    padding: 0.9rem 1.8rem;
    border-right: 1px solid rgba(0,0,0,0.08);
}
.kpi:last-child { border-right: none; }
.kpi-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem; font-weight: 700; color: #0d1b2a; line-height: 1;
}
.kpi-label {
    font-size: 0.68rem; color: rgba(0,0,0,0.45);
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px;
}

.glass {
    background: rgba(0,0,0,0.02);
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px; padding: 1.8rem;
}
.glass-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.95rem; font-weight: 600;
    color: rgba(0,0,0,0.6);
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 1.4rem; padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}

.stNumberInput label, .stRadio label,
.stSelectbox label, .stFileUploader label {
    font-size: 0.72rem !important;
    color: rgba(0,0,0,0.5) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    font-family: 'Syne', sans-serif !important;
}
.stNumberInput input, .stSelectbox select {
    background: #f5f5f5 !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stRadio"] > div { gap: 10px !important; flex-direction: row !important; }
[data-testid="stRadio"] label {
    background: #f5f5f5 !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important; padding: 9px 20px !important;
    cursor: pointer !important; transition: all .2s !important;
    text-transform: none !important; letter-spacing: normal !important;
    font-size: 0.85rem !important; color: rgba(0,0,0,0.65) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(26,111,255,0.1) !important;
    border-color: #1a6fff !important;
    color: #1a6fff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a6fff, #0052d9) !important;
    color: #fff !important; border: none !important;
    border-radius: 50px !important; padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: 0.85rem !important; letter-spacing: 0.05em !important;
    transition: all .25s !important;
    box-shadow: 0 4px 20px rgba(26,111,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(26,111,255,0.5) !important;
}
.stDownloadButton > button {
    background: #f5f5f5 !important;
    color: rgba(0,0,0,0.55) !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 50px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
}
.stDownloadButton > button:hover {
    border-color: #1a6fff !important; color: #1a6fff !important;
}

[data-testid="stFileUploader"] {
    background: #fafafa !important;
    border: 2px dashed rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; }

.result-card {
    border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem;
    border: 1px solid; position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
}
.result-low  { background: rgba(0,201,167,0.05); border-color: rgba(0,201,167,0.3); }
.result-low::before  { background: linear-gradient(90deg, transparent, #00c9a7, transparent); }
.result-med  { background: rgba(255,187,0,0.05);  border-color: rgba(255,187,0,0.3); }
.result-med::before  { background: linear-gradient(90deg, transparent, #ffbb00, transparent); }
.result-high { background: rgba(255,75,75,0.05);  border-color: rgba(255,75,75,0.3); }
.result-high::before { background: linear-gradient(90deg, transparent, #ff4b4b, transparent); }

.result-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 50px;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1.2rem;
}
.badge-low  { background: rgba(0,201,167,0.12);  color: #00a87e; border: 1px solid rgba(0,201,167,0.3); }
.badge-med  { background: rgba(255,187,0,0.12);  color: #cc9500; border: 1px solid rgba(255,187,0,0.3); }
.badge-high { background: rgba(255,75,75,0.12);  color: #cc3b3b; border: 1px solid rgba(255,75,75,0.3); }

.result-prob {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.5rem; font-weight: 700; line-height: 1; margin-bottom: 0.3rem;
}
.prob-low  { color: #00a87e; }
.prob-med  { color: #cc9500; }
.prob-high { color: #cc3b3b; }

.result-sublabel {
    font-size: 0.72rem; color: rgba(0,0,0,0.45);
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.4rem;
}
.result-decision {
    background: rgba(0,0,0,0.02);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.85rem; color: rgba(0,0,0,0.6); line-height: 1.7;
}
.result-decision strong {
    color: #0d1b2a; display: block;
    font-size: 0.72rem; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 5px;
}

.stProgress > div > div > div {
    border-radius: 50px !important; height: 5px !important;
    background: rgba(0,0,0,0.05) !important;
}
.stProgress > div > div > div > div { border-radius: 50px !important; }

.mcard {
    background: #fafafa;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px; padding: 1.6rem;
    margin-bottom: 1rem;
}
.mcard-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem; font-weight: 600;
    color: #0d1b2a;
    margin-bottom: 1.1rem; padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}
.mcard p, .mcard li {
    font-size: 0.85rem; color: rgba(0,0,0,0.6);
    line-height: 1.7; margin-bottom: 0.5rem;
}
.mcard strong { color: #0d1b2a; font-weight: 600; }
.mcard ul { padding-left: 1.2rem; }

.step-row {
    display: flex; gap: 12px; margin-bottom: 1rem;
    font-size: 0.85rem; color: rgba(0,0,0,0.55); line-height: 1.65;
}
.step-dot {
    width: 24px; height: 24px; border-radius: 50%;
    background: rgba(26,111,255,0.1);
    border: 1px solid rgba(26,111,255,0.4);
    color: #1a6fff; font-size: 0.7rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.formula-box {
    background: rgba(26,111,255,0.08);
    border: 1px solid rgba(26,111,255,0.2);
    border-radius: 10px; padding: 1.1rem; text-align: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem; color: #1a6fff;
    letter-spacing: 0.08em; margin-top: 1rem;
}
.threshold-row {
    display: flex; align-items: flex-start; gap: 12px; margin-bottom: 0.9rem;
}
.th-ind { width: 4px; height: 40px; border-radius: 2px; flex-shrink: 0; }
.th-low  { background: #00c9a7; }
.th-med  { background: #ffbb00; }
.th-high { background: #ff4b4b; }
.th-title { color: #0d1b2a; font-weight: 600; font-size: 0.85rem; }
.th-desc  { color: rgba(0,0,0,0.5); font-size: 0.78rem; margin-top: 4px; }

.placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 380px;
    background: #fafafa;
    border: 1px dashed rgba(0,0,0,0.12);
    border-radius: 16px; color: rgba(0,0,0,0.35);
    font-size: 0.85rem; text-align: center; line-height: 1.7; gap: 1rem;
}

.contact-card {
    background: #fafafa;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.contact-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}
.contact-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.contact-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0d1b2a;
    margin-bottom: 0.25rem;
}
.contact-detail {
    font-size: 0.85rem;
    color: #1a6fff;
    text-decoration: none;
}
.contact-desc {
    font-size: 0.75rem;
    color: rgba(0,0,0,0.45);
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(current_dir, "model_multimodal_54.pkl")
    json_path   = os.path.join(current_dir, "params_multimodal_54.json")
    try:
        mdl = joblib.load(model_path)
        with open(json_path, 'r') as f:
            prms = json.load(f)
        return mdl, prms
    except Exception:
        return None, None

model, params = load_assets()
model_ok = model is not None

# ── TOPBAR NAVIGATION AVEC BOUTONS FONCTIONNELS ────────────────────────────────
# Déterminer la classe active pour chaque lien
active_class = lambda page: "active" if st.session_state['current_page'] == page else ""

st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🧬</div>
        <span class="topbar-name">MelanomaPredict AI</span>
    </div>
    <div class="nav-container">
        <button class="nav-link {active_class('analyse')}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'analyse'}}, '*')">ANALYSE</button>
        <span class="nav-separator">|</span>
        <button class="nav-link {active_class('methodologie')}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'methodologie'}}, '*')">MÉTHODOLOGIE</button>
        <span class="nav-separator">|</span>
        <button class="nav-link {active_class('documentation')}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'documentation'}}, '*')">DOCUMENTATION</button>
        <span class="nav-separator">|</span>
        <button class="nav-link {active_class('contact')}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'contact'}}, '*')">CONTACT</button>
    </div>
    <div class="topbar-status">
        <span class="pulse"></span>&nbsp; Système opérationnel
    </div>
</div>
""", unsafe_allow_html=True)

# Capture des clics via les colonnes Streamlit
nav_cols = st.columns(4)
nav_pages = ["analyse", "methodologie", "documentation", "contact"]
nav_labels = ["ANALYSE", "MÉTHODOLOGIE", "DOCUMENTATION", "CONTACT"]

for i, (col, page, label) in enumerate(zip(nav_cols, nav_pages, nav_labels)):
    with col:
        if st.button(label, key=f"nav_{page}", use_container_width=True):
            navigate_to(page)

st.divider()
# ── CONTENU DES PAGES ─────────────────────────────────────────────────────────

# PAGE ANALYSE
if st.session_state['current_page'] == 'analyse':
    # Hero section
    st.markdown("""
    <div class="hero">
        <div class="hero-img"></div>
        <div class="hero-gradient"></div>
        <div class="hero-content">
            <div class="hero-eyebrow">
                <span class="hero-eyebrow-dot"></span>
                DISPOSITIF DE RECHERCHE CLINIQUE
            </div>
            <h1>Analyse Multimodale<br>du <span>Mélanome</span> Cutané</h1>
            <p class="hero-sub">
                Aide à la décision thérapeutique par analyse combinée de
                54 biomarqueurs transcriptomiques et paramètres cliniques.
                Cohorte de référence TCGA-SKCM.
            </p>
            <div class="hero-kpis">
                <div class="kpi">
                    <div class="kpi-val">54</div>
                    <div class="kpi-label">GÈNES ANALYSÉS</div>
                </div>
                <div class="kpi">
                    <div class="kpi-val">500</div>
                    <div class="kpi-label">ARBRES DÉCISIONNELS</div>
                </div>
                <div class="kpi">
                    <div class="kpi-val">TCGA</div>
                    <div class="kpi-label">COHORTE DE RÉFÉRENCE</div>
                </div>
                <div class="kpi">
                    <div class="kpi-val">57</div>
                    <div class="kpi-label">FEATURES TOTALES</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning(
        "**Dispositif de Recherche.** Ce système génère un score de risque basé sur "
        "54 signatures transcriptomiques. Il ne remplace pas le jugement clinique d'un médecin."
    )
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([5, 7], gap="large")

    with col_in:
        st.markdown('<div class="glass"><div class="glass-title"> &nbsp;PARAMÈTRES CLINIQUES</div>', unsafe_allow_html=True)
        age   = st.number_input("Âge du patient", min_value=1, max_value=115, value=55)
        sexe  = st.radio("Sexe biologique", ["Homme", "Femme"], horizontal=True)
        stade = st.selectbox("Stade TNM initial", ["I", "II", "III", "IV"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        st.markdown('<div class="glass"><div class="glass-title">🔬 &nbsp;DONNÉES OMIQUES</div>', unsafe_allow_html=True)
        if params:
            example_df = pd.DataFrame(
                np.random.uniform(0.5, 5.0, size=(1, 54)),
                columns=params['top_genes']
            )
            st.download_button(
                label="↓  Télécharger le template .csv",
                data=example_df.to_csv(index=False).encode('utf-8'),
                file_name="template_54_genes.csv",
                mime="text/csv",
                use_container_width=True
            )
        uploaded_file = st.file_uploader(
            "Profil d'expression génique — 54 gènes (HGNC Symbols)",
            type="csv"
        )
        run_btn = st.button(
            "Lancer l'analyse diagnostique →",
            use_container_width=True,
            disabled=(uploaded_file is None or not model_ok)
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        if run_btn and uploaded_file and model and params:
            df_patient = pd.read_csv(uploaded_file)
            with st.spinner("Analyse en cours…"):
                sexe_val   = 0 if sexe == "Homme" else 1
                stade_val  = {"I": 1, "II": 2, "III": 3, "IV": 4}[stade]
                omique_vec = df_patient[params['top_genes']].iloc[0].tolist()
                X          = np.array([age, sexe_val, stade_val] + omique_vec).reshape(1, -1)
                X_scaled   = (X - np.array(params['means'])) / np.array(params['stds'])
                prob       = model.predict_proba(X_scaled)[0][1]
                st.session_state['analysis'] = {'prob': prob, 'genes': params['top_genes']}

        if 'analysis' in st.session_state:
            res  = st.session_state['analysis']
            prob = res['prob']
            pct  = prob * 100

            if prob < 0.33:
                rc, bc, pc = "result-low",  "badge-low",  "prob-low"
                badge_txt  = "● &nbsp;RISQUE FAIBLE"
                decision   = ("<strong>Recommandation</strong><br>"
                              "Mélanome primaire probable. Surveillance standard et "
                              "suivi dermatologique classique recommandé.")
                bar_color  = "#00c9a7"
            elif prob < 0.67:
                rc, bc, pc = "result-med",  "badge-med",  "prob-med"
                badge_txt  = "● &nbsp;RISQUE INTERMÉDIAIRE"
                decision   = ("<strong>Recommandation</strong><br>"
                              "Zone d'incertitude clinique. Examens complémentaires, "
                              "confirmation histologique et suivi rapproché.")
                bar_color  = "#ffbb00"
            else:
                rc, bc, pc = "result-high", "badge-high", "prob-high"
                badge_txt  = "● &nbsp;RISQUE ÉLEVÉ"
                decision   = ("<strong>Recommandation</strong><br>"
                              "Mélanome métastatique probable. Discussion précoce "
                              "d'immunothérapie (anti-PD-1 / anti-CTLA-4) ou thérapie ciblée.")
                bar_color  = "#ff4b4b"

            st.markdown(
                f'<div class="result-card {rc}">'
                f'<div class="result-badge {bc}">{badge_txt}</div>'
                f'<div class="result-prob {pc}">{pct:.1f}<span style="font-size:2rem">%</span></div>'
                f'<div class="result-sublabel">Probabilité Métastatique — Score p</div>'
                f'<div class="result-decision">{decision}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.progress(prob)

            # Biomarker chart
            importances = model.feature_importances_[3:]
            df_imp = (
                pd.DataFrame({'gene': res['genes'], 'imp': importances})
                .sort_values('imp', ascending=True)
                .tail(10)
            )
            fig = go.Figure(go.Bar(
                x=df_imp['imp'], y=df_imp['gene'], orientation='h',
                marker=dict(
                    color=df_imp['imp'],
                    colorscale=[[0, "rgba(0,0,0,0.04)"], [1, bar_color]],
                    line=dict(width=0)
                ),
                hovertemplate='<b>%{y}</b><br>Importance : %{x:.4f}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(
                    text="Top 10 — Biomarqueurs Décisifs",
                    font=dict(family="Cormorant Garamond, serif", size=17, color="#0d1b2a"),
                    x=0
                ),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                    color="rgba(0,0,0,0.45)",
                    tickfont=dict(family="Syne, sans-serif", size=10), zeroline=False
                ),
                yaxis=dict(
                    color="#0d1b2a",
                    tickfont=dict(family="Syne, sans-serif", size=11),
                    gridcolor="rgba(0,0,0,0)"
                ),
                margin=dict(l=0, r=0, t=45, b=0), height=340
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        else:
            st.markdown("""
            <div class="placeholder">
                <div style="font-size:2.5rem; opacity:0.3;">🧬</div>
                <div>
                    Chargez un profil d'expression génique<br>
                    puis lancez le diagnostic pour afficher les résultats.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE MÉTHODOLOGIE — remplace le bloc elif correspondant
# ════════════════════════════════════════════════════════
METHODOLOGIE = '''
elif st.session_state['current_page'] == 'methodologie':
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow"><span class="page-eyebrow-line"></span>Architecture Scientifique</div>
        <h1 class="page-title">Méthodologie</h1>
        <p class="page-subtitle">Protocole de conception, validation et déploiement du modèle prédictif multimodal.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(\'<div class="section-label">01 — Architecture du Modèle</div>\', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="arch-block">
            <div class="arch-block-header">
                <div class="arch-icon arch-icon-blue">🧠</div>
                <div class="arch-block-title">Moteur Prédictif</div>
                <span class="arch-block-tag">Core</span>
            </div>
            <div class="arch-block-body">
                <p>Random Forest de <strong>500 arbres décisionnels</strong> entraîné sur la cohorte TCGA-SKCM. Chaque arbre vote indépendamment ; la probabilité finale est la moyenne agrégée des votes foliaires.</p>
                <div class="arch-chip-row">
                    <span class="arch-chip">500 estimators</span>
                    <span class="arch-chip">max_features = sqrt</span>
                    <span class="arch-chip">Gini impurity</span>
                    <span class="arch-chip">OOB scoring</span>
                </div>
            </div>
        </div>
        <div class="arch-block">
            <div class="arch-block-header">
                <div class="arch-icon arch-icon-teal">🔬</div>
                <div class="arch-block-title">Signature Génomique</div>
                <span class="arch-block-tag">54 gènes</span>
            </div>
            <div class="arch-block-body">
                <p>54 biomarqueurs mRNA sélectionnés par <strong>régression Lasso pénalisée</strong> sur l'ensemble du transcriptome TCGA. Couvrent l'invasion tumorale, le remodelage de la MEC, la transition EMT et l'inflammation.</p>
                <div class="arch-chip-row">
                    <span class="arch-chip">Lasso α-optimisé</span>
                    <span class="arch-chip">Log2 TPM</span>
                    <span class="arch-chip">HGNC Symbols</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="arch-block">
            <div class="arch-block-header">
                <div class="arch-icon arch-icon-amber">📊</div>
                <div class="arch-block-title">Normalisation Z-score</div>
                <span class="arch-block-tag">Preprocessing</span>
            </div>
            <div class="arch-block-body">
                <p>Chaque variable subit une standardisation basée sur les paramètres de la cohorte TCGA, assurant la comparabilité inter-laboratoires indépendamment du pipeline RNA-seq utilisé.</p>
                <div class="formula-elegant">
                    <div class="formula-math">z = (x − μ) / σ</div>
                    <div class="formula-sub">μ, σ calculés sur TCGA-SKCM (n = 473)</div>
                </div>
            </div>
        </div>
        <div class="arch-block">
            <div class="arch-block-header">
                <div class="arch-icon arch-icon-rose">✅</div>
                <div class="arch-block-title">Validation Croisée</div>
                <span class="arch-block-tag">10-Fold CV</span>
            </div>
            <div class="arch-block-body">
                <p>Validation croisée <strong>stratifiée à 10 folds</strong> préservant la distribution des classes. Métriques reportées : moyennes ± écarts-types sur les 10 partitions.</p>
                <div class="arch-chip-row">
                    <span class="arch-chip">Stratified K-Fold</span>
                    <span class="arch-chip">Train 80 / Test 20</span>
                    <span class="arch-chip">Balanced classes</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(\'<div class="section-label">02 — Pipeline Diagnostique</div>\', unsafe_allow_html=True)
    c3, c4 = st.columns([5, 7], gap="large")

    with c3:
        st.markdown("""
        <div class="pipeline-wrap">
            <div class="pipeline-title">🔄 Flux de Traitement</div>
            <div class="pipe-step">
                <div class="pipe-num">01</div>
                <div class="pipe-content">
                    <div class="pipe-label">Acquisition</div>
                    <div class="pipe-desc">Saisie des paramètres cliniques (Âge, Sexe, Stade TNM) et chargement du profil d'expression au format CSV normalisé HGNC.</div>
                </div>
            </div>
            <div class="pipe-step">
                <div class="pipe-num">02</div>
                <div class="pipe-content">
                    <div class="pipe-label">Fusion Multimodale</div>
                    <div class="pipe-desc">Encodage binaire du sexe, ordinalisation du stade TNM, concaténation en vecteur unique de <strong>57 variables</strong> (54G + 3C).</div>
                </div>
            </div>
            <div class="pipe-step">
                <div class="pipe-num">03</div>
                <div class="pipe-content">
                    <div class="pipe-label">Standardisation</div>
                    <div class="pipe-desc">Application des paramètres μ/σ TCGA sur chaque variable. Alignement sur l'espace de référence de la cohorte.</div>
                </div>
            </div>
            <div class="pipe-step">
                <div class="pipe-num">04</div>
                <div class="pipe-content">
                    <div class="pipe-label">Inférence RF</div>
                    <div class="pipe-desc">Vote agrégé des 500 arbres. Calcul de <em>p</em> métastatique par moyenne des probabilités foliaires pondérées.</div>
                </div>
            </div>
            <div class="pipe-step" style="margin-bottom:0">
                <div class="pipe-num">05</div>
                <div class="pipe-content">
                    <div class="pipe-label">Interprétation</div>
                    <div class="pipe-desc">Seuillage tripartite et visualisation du top-10 des biomarqueurs décisifs par feature importance normalisée.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="pipeline-wrap">
            <div class="pipeline-title">🎯 Seuils de Décision Clinique</div>
            <div class="thr-grid">
                <div class="thr-card thr-card-low">
                    <div class="thr-pct thr-pct-low">&lt;33<span style="font-size:1rem">%</span></div>
                    <div class="thr-divider"></div>
                    <div>
                        <div class="thr-title">Risque Faible</div>
                        <div class="thr-desc">Mélanome primaire probable. Surveillance standard et suivi dermatologique classique recommandé.</div>
                    </div>
                </div>
                <div class="thr-card thr-card-med">
                    <div class="thr-pct thr-pct-med">33–67<span style="font-size:1rem">%</span></div>
                    <div class="thr-divider"></div>
                    <div>
                        <div class="thr-title">Risque Intermédiaire</div>
                        <div class="thr-desc">Zone d'incertitude clinique. Confirmation histologique et imagerie complémentaire requises.</div>
                    </div>
                </div>
                <div class="thr-card thr-card-high">
                    <div class="thr-pct thr-pct-high">≥67<span style="font-size:1rem">%</span></div>
                    <div class="thr-divider"></div>
                    <div>
                        <div class="thr-title">Risque Élevé</div>
                        <div class="thr-desc">Mélanome métastatique probable. Discussion précoce anti-PD-1 / anti-CTLA-4 ou thérapie ciblée BRAF/MEK.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
'''



# ════════════════════════════════════════════════════════
# PAGE DOCUMENTATION — remplace le bloc elif correspondant
# ════════════════════════════════════════════════════════
DOCUMENTATION = '''
elif st.session_state['current_page'] == 'documentation':
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow"><span class="page-eyebrow-line"></span>Référence Technique</div>
        <h1 class="page-title">Documentation Scientifique</h1>
        <p class="page-subtitle">Modèle multimodal pour la prédiction du risque métastatique dans le mélanome cutané (TCGA-SKCM).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-strip">
        <div class="stat-cell">
            <div class="stat-cell-val">473</div>
            <div class="stat-cell-label">Patients TCGA</div>
        </div>
        <div class="stat-cell">
            <div class="stat-cell-val">54</div>
            <div class="stat-cell-label">Gènes sélectionnés</div>
        </div>
        <div class="stat-cell">
            <div class="stat-cell-val">0.955</div>
            <div class="stat-cell-label">AUC-ROC</div>
        </div>
        <div class="stat-cell">
            <div class="stat-cell-val">90<span>%</span></div>
            <div class="stat-cell-label">Accuracy globale</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([6, 6], gap="large")

    with col1:
        st.markdown(\'<div class="section-label">Performances du Modèle</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="perf-wrap">
            <div class="perf-row">
                <div class="perf-meta"><span class="perf-metric-name">Accuracy</span><span class="perf-metric-val">90.0 %</span></div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:90%"></div></div>
            </div>
            <div class="perf-row">
                <div class="perf-meta"><span class="perf-metric-name">Sensibilité (Recall)</span><span class="perf-metric-val">85.0 %</span></div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:85%"></div></div>
            </div>
            <div class="perf-row">
                <div class="perf-meta"><span class="perf-metric-name">Spécificité</span><span class="perf-metric-val">95.0 %</span></div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:95%"></div></div>
            </div>
            <div class="perf-row">
                <div class="perf-meta"><span class="perf-metric-name">AUC-ROC</span><span class="perf-metric-val">0.955</span></div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:95.5%"></div></div>
            </div>
            <div class="perf-row">
                <div class="perf-meta"><span class="perf-metric-name">F1-Score</span><span class="perf-metric-val">89.47 %</span></div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:89.47%"></div></div>
            </div>
            <div class="perf-row">
                <div class="perf-meta"><span class="perf-metric-name">Precision</span><span class="perf-metric-val">94.44 %</span></div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:94.44%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(\'<div class="section-label">Limitations & Usage</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="gene-panel">
            <div class="gene-panel-title">⚠️ Limitations connues</div>
            <div class="limit-item">
                <div class="limit-num">1</div>
                <div class="limit-text"><strong>Validation externe en cours</strong> sur cohortes indépendantes GEO et ICGC. Performances à confirmer hors cadre TCGA.</div>
            </div>
            <div class="limit-item">
                <div class="limit-num">2</div>
                <div class="limit-text"><strong>Ne remplace pas le gold standard</strong> histologique — diagnostic anatomopathologique toujours requis.</div>
            </div>
            <div class="limit-item">
                <div class="limit-num">3</div>
                <div class="limit-text"><strong>Usage réservé à la recherche clinique.</strong> Non certifié comme dispositif médical (CE / FDA).</div>
            </div>
            <div class="limit-item">
                <div class="limit-num">4</div>
                <div class="limit-text"><strong>Normalisation stricte requise</strong> — expressions géniques produites par RNA-seq TPM et normalisées en log2(x+1).</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(\'<div class="section-label">Base de Données & Cohorte</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-block">
            <div class="arch-block-header">
                <div class="arch-icon arch-icon-blue">🗄️</div>
                <div class="arch-block-title">TCGA-SKCM</div>
                <span class="arch-block-tag">Référence</span>
            </div>
            <div class="arch-block-body">
                <p><strong>The Cancer Genome Atlas – Skin Cutaneous Melanoma.</strong> Cohorte multicentrique incluant 473 patients avec données RNA-seq complètes et annotation clinique détaillée.</p>
                <div class="arch-chip-row">
                    <span class="arch-chip">RNA-seq FPKM</span>
                    <span class="arch-chip">473 patients</span>
                    <span class="arch-chip">Primaire + Métastatique</span>
                    <span class="arch-chip">Annoté TNM</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(\'<div class="section-label" style="margin-top:1.5rem">Signature Génomique — 54 Biomarqueurs</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="gene-panel">
            <div class="gene-panel-title">🧬 Gènes sélectionnés par Lasso</div>
            <div class="gene-category">Invasion & Remodelage MEC</div>
            <div class="gene-tag-grid">
                <span class="gene-tag">MMP1</span><span class="gene-tag">MMP9</span>
                <span class="gene-tag">MMP14</span><span class="gene-tag">COL1A1</span>
                <span class="gene-tag">COL3A1</span><span class="gene-tag">LAMC2</span>
                <span class="gene-tag">FN1</span><span class="gene-tag">ITGAV</span>
            </div>
            <div class="gene-category">Transition Épithélio-Mésenchymateuse (EMT)</div>
            <div class="gene-tag-grid">
                <span class="gene-tag">CDH1</span><span class="gene-tag">CDH2</span>
                <span class="gene-tag">VIM</span><span class="gene-tag">ZEB1</span>
                <span class="gene-tag">ZEB2</span><span class="gene-tag">TWIST1</span>
                <span class="gene-tag">SNAI1</span><span class="gene-tag">SNAI2</span>
            </div>
            <div class="gene-category">Immunité & Checkpoints</div>
            <div class="gene-tag-grid">
                <span class="gene-tag">PDCD1</span><span class="gene-tag">CD274</span>
                <span class="gene-tag">CTLA4</span><span class="gene-tag">LAG3</span>
                <span class="gene-tag">TIGIT</span><span class="gene-tag">HAVCR2</span>
                <span class="gene-tag">CD8A</span><span class="gene-tag">FOXP3</span>
            </div>
            <div class="gene-category">Prolifération & Cycle Cellulaire</div>
            <div class="gene-tag-grid">
                <span class="gene-tag">BRAF</span><span class="gene-tag">NRAS</span>
                <span class="gene-tag">CDKN2A</span><span class="gene-tag">TP53</span>
                <span class="gene-tag">PTEN</span><span class="gene-tag">MKI67</span>
                <span class="gene-tag">CCND1</span><span class="gene-tag">CDK4</span>
            </div>
            <p style="font-size:0.72rem;color:rgba(0,0,0,0.3);margin-top:1rem;margin-bottom:0">
                + 22 biomarqueurs additionnels (angiogenèse, métabolisme, signalisation MAPK/PI3K)
            </p>
        </div>
        """, unsafe_allow_html=True)
'''



# ════════════════════════════════════════════════════════
# PAGE CONTACT — remplace le bloc elif correspondant
# ════════════════════════════════════════════════════════
CONTACT = '''
elif st.session_state['current_page'] == 'contact':
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow"><span class="page-eyebrow-line"></span>Support & Collaboration</div>
        <h1 class="page-title">Contact</h1>
        <p class="page-subtitle">Une question scientifique, une demande de collaboration ou un support technique ? L'équipe répond sous 48 h.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([6, 6], gap="large")

    with c1:
        st.markdown("""
        <div class="cta-strip">
            <div class="cta-overline">Prise de Contact</div>
            <h2 class="cta-title">Travaillons<br>Ensemble</h2>
            <p class="cta-sub">Collaborations cliniques, intégrations hospitalières, demandes de licences de recherche ou support technique.</p>
            <a class="cta-email" href="mailto:contact@melanomapredict.ai">
                ✉&nbsp; contact@melanomapredict.ai
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="contact-grid">
            <div class="cont-card">
                <div class="cont-icon">🔬</div>
                <div class="cont-label">Recherche</div>
                <div class="cont-value">Collaborations Cliniques</div>
                <div class="cont-desc">Intégration dans des cohortes de validation, co-publications, études prospectives.</div>
            </div>
            <div class="cont-card">
                <div class="cont-icon">🏥</div>
                <div class="cont-label">Institutionnel</div>
                <div class="cont-value">Déploiement Hospitalier</div>
                <div class="cont-desc">Intégration SIH, accompagnement réglementaire, formation des équipes médicales.</div>
            </div>
            <div class="cont-card">
                <div class="cont-icon">⚙️</div>
                <div class="cont-label">Technique</div>
                <div class="cont-value">Support & API</div>
                <div class="cont-desc">Documentation technique, accès API, normalisation des données d'expression génique.</div>
            </div>
            <div class="cont-card">
                <div class="cont-icon">📄</div>
                <div class="cont-label">Réglementaire</div>
                <div class="cont-value">Licences & Usage</div>
                <div class="cont-desc">Licences de recherche, conformité RGPD, cadre d'utilisation clinique.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(\'<div class="section-label" style="margin-top:0">FAQ</div>\', unsafe_allow_html=True)
        st.markdown("""
        <div class="gene-panel">
            <div class="gene-panel-title">💬 Questions Fréquentes</div>
            <div class="faq-item">
                <div class="faq-q">Quel format de données d'expression génique est requis ?</div>
                <div class="faq-a">Les données doivent être au format RNA-seq, normalisées en TPM ou FPKM puis transformées en log2(x+1). Un template CSV téléchargeable est disponible sur la page Analyse.</div>
            </div>
            <div class="faq-item">
                <div class="faq-q">Le modèle est-il validé sur d'autres cohortes que TCGA ?</div>
                <div class="faq-a">La validation externe est en cours sur les cohortes GEO et ICGC. Les résultats préliminaires seront publiés prochainement.</div>
            </div>
            <div class="faq-item">
                <div class="faq-q">Peut-on l'utiliser pour d'autres types tumoraux que le mélanome cutané ?</div>
                <div class="faq-a">Non. Le modèle a été spécifiquement entraîné sur TCGA-SKCM. Son application à d'autres tumeurs n'est pas validée et déconseillée.</div>
            </div>
            <div class="faq-item">
                <div class="faq-q">Quelle est la politique de confidentialité des données patients ?</div>
                <div class="faq-a">Aucune donnée patient n'est transmise à des serveurs tiers. Tout le traitement est réalisé localement dans l'instance Streamlit déployée.</div>
            </div>
            <div class="faq-item" style="border-bottom:none;padding-bottom:0">
                <div class="faq-q">Comment citer cet outil dans une publication ?</div>
                <div class="faq-a">Contactez-nous à l'adresse ci-contre pour recevoir la référence bibliographique complète et les conditions de citation.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
'''
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem; text-align:center;
                border-bottom:1px solid rgba(0,0,0,0.08); margin-bottom:1.2rem;">
        <div style="font-size:2rem; margin-bottom:0.5rem;">🧬</div>
        <div style="font-family:'Cormorant Garamond',serif;
                    font-size:1.05rem; font-weight:700; color:#0d1b2a;">
            MelanomaPredict AI
        </div>
        <div style="font-size:0.65rem; color:rgba(0,0,0,0.4);
                    letter-spacing:0.12em; text-transform:uppercase; margin-top:3px;">
            Portail Diagnostic v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation rapide**")
    
    if st.button("🚀 Analyse", use_container_width=True, key="sidebar_analyse"):
        st.session_state['current_page'] = 'analyse'
        st.rerun()
    
    if st.button("📖 Méthodologie", use_container_width=True, key="sidebar_methodo"):
        st.session_state['current_page'] = 'methodologie'
        st.rerun()
    
    if st.button("📚 Documentation", use_container_width=True, key="sidebar_doc"):
        st.session_state['current_page'] = 'documentation'
        st.rerun()
    
    if st.button("📧 Contact", use_container_width=True, key="sidebar_contact"):
        st.session_state['current_page'] = 'contact'
        st.rerun()
    
    st.divider()
    
    st.markdown("**Statut Système**")
    if model_ok:
        st.success("✅ Modèle chargé avec succès")
        st.info("🌲 Random Forest · 500 arbres")
        st.info("🔬 57 features · 54 gènes + 3 cliniques")
    else:
        st.error("❌ Modèle introuvable")
        st.warning("Vérifiez `model_multimodal_54.pkl` et `params_multimodal_54.json` dans le répertoire")

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem; color:rgba(0,0,0,0.35);
                line-height:1.6; padding:0 0.2rem;">
        <strong>⚠️ Avertissement</strong><br>
        Cet outil est réservé à un usage de recherche. Il ne constitue pas
        un dispositif médical certifié et ne remplace pas l'avis d'un
        professionnel de santé qualifié.
    </div>
    """, unsafe_allow_html=True)

