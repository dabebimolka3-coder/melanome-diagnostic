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
* { font-family: 'Syne', sans-serif; color: #1a1a2e; }
h1, h2, h3, h4, h5, h6 { font-family: 'Cormorant Garamond', serif !important; color: #0d1b2a !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.topbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(0,0,0,0.08);
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem; height: 62px;
    margin: 0 -2.5rem 0;
}
.topbar-brand { display: flex; align-items: center; gap: 10px; }
.topbar-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1a6fff, #00c9a7);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    color: white;
}
.topbar-name {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.15rem; font-weight: 700;
    color: #0d1b2a; letter-spacing: 0.03em;
}
.topbar-links { display: flex; gap: 2.5rem; align-items: center; }
.topbar-links a {
    font-size: 0.78rem; color: rgba(0,0,0,0.5);
    text-decoration: none; letter-spacing: 0.08em;
    text-transform: uppercase; font-weight: 600;
}
.topbar-links a:hover { color: #1a6fff; }
.topbar-status {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.72rem; color: rgba(0,0,0,0.45);
    letter-spacing: 0.06em;
}
.pulse {
    width: 7px; height: 7px; border-radius: 50%;
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
    position: relative; min-height: 420px;
    margin: 0 -2.5rem 3rem;
    overflow: hidden; display: flex; align-items: flex-end;
    border-radius: 0;
}
.hero-img {
    position: absolute; inset: 0;
    background-image: url('https://raw.githubusercontent.com/mauhong/melanoma-detection/main/images/melanoma_skin.jpg');
    background-size: cover;
    background-position: center 30%;
    background-repeat: no-repeat;
    filter: brightness(0.75) contrast(1.1) saturate(1.15);
    transform: scale(1.02);
}
.hero-img::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.1) 100%);
}
.hero-gradient {
    position: absolute; inset: 0;
    background: linear-gradient(to top, #ffffff 0%, rgba(255,255,255,0.85) 40%, rgba(255,255,255,0.3) 100%);
}
.hero-content {
    position: relative; z-index: 2;
    padding: 0 3rem 3rem; max-width: 780px;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(26,111,255,0.3);
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
    font-size: 3.2rem !important; font-weight: 700 !important;
    color: #ffffff !important; line-height: 1.1 !important;
    margin: 0 0 1rem !important; letter-spacing: -0.01em;
    text-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.hero h1 span { color: #00c9a7; text-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.hero-sub {
    color: rgba(255,255,255,0.9); font-size: 0.95rem;
    line-height: 1.7; max-width: 520px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.15);
    font-weight: 500;
}
.hero-kpis {
    display: flex; gap: 0; margin-top: 2rem;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 12px; overflow: hidden; width: fit-content;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
}
.kpi {
    padding: 0.9rem 1.8rem;
    border-right: 1px solid rgba(255,255,255,0.2);
}
.kpi:last-child { border-right: none; }
.kpi-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem; font-weight: 700; color: #ffffff; line-height: 1;
    text-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
.kpi-label {
    font-size: 0.68rem; color: rgba(255,255,255,0.85);
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

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #f5f5f5 !important;
    border-radius: 50px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    padding: 4px !important; width: fit-content !important;
    gap: 2px !important; margin-bottom: 2.5rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 50px !important; padding: 9px 26px !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 600 !important; color: rgba(0,0,0,0.45) !important;
    background: transparent !important; border: none !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(26,111,255,0.1) !important;
    color: #1a6fff !important;
    border: 1px solid rgba(26,111,255,0.25) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

[data-testid="stSidebar"] {
    background: #fafafa !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}
[data-testid="stSidebar"] * { color: rgba(0,0,0,0.65) !important; }

.stWarning > div {
    background: rgba(255,187,0,0.08) !important;
    border: 1px solid rgba(255,187,0,0.25) !important;
    border-radius: 12px !important; color: #a67c00 !important;
    font-size: 0.82rem !important;
}
.stSuccess > div {
    background: rgba(0,201,167,0.08) !important;
    border: 1px solid rgba(0,201,167,0.25) !important; border-radius: 10px !important;
}
.stInfo > div {
    background: rgba(26,111,255,0.08) !important;
    border: 1px solid rgba(26,111,255,0.25) !important; border-radius: 10px !important;
}
.stError > div {
    background: rgba(255,75,75,0.08) !important;
    border: 1px solid rgba(255,75,75,0.25) !important; border-radius: 10px !important;
}
hr { border-color: rgba(0,0,0,0.08) !important; }

.mcard {
    background: #fafafa;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px; padding: 1.6rem;
}
.mcard-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem; font-weight: 600;
    color: #0d1b2a;
    margin-bottom: 1.1rem; padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}
.mcard p, .mcard li {
    font-size: 0.83rem; color: rgba(0,0,0,0.55);
    line-height: 1.75; margin-bottom: 0.5rem;
}
.mcard strong { color: #0d1b2a; font-weight: 600; }
.mcard ul { padding-left: 1.2rem; }

.step-row {
    display: flex; gap: 12px; margin-bottom: 1rem;
    font-size: 0.83rem; color: rgba(0,0,0,0.5); line-height: 1.65;
}
.step-dot {
    width: 22px; height: 22px; border-radius: 50%;
    background: rgba(26,111,255,0.1);
    border: 1px solid rgba(26,111,255,0.4);
    color: #1a6fff; font-size: 0.68rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
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
.th-ind { width: 3px; height: 36px; border-radius: 2px; flex-shrink: 0; margin-top: 2px; }
.th-low  { background: #00c9a7; }
.th-med  { background: #ffbb00; }
.th-high { background: #ff4b4b; }
.th-title { color: #0d1b2a; font-weight: 600; font-size: 0.82rem; }
.th-desc  { color: rgba(0,0,0,0.5); font-size: 0.78rem; margin-top: 2px; }

.placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 380px;
    background: #fafafa;
    border: 1px dashed rgba(0,0,0,0.12);
    border-radius: 16px; color: rgba(0,0,0,0.35);
    font-size: 0.85rem; text-align: center; line-height: 1.7; gap: 1rem;
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

# ── TOP NAV ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🧬</div>
        <span class="topbar-name">MelanomaPredict AI</span>
    </div>
    <div class="topbar-links">
        <a href="#">Analyse</a>
        <a href="#">Méthodologie</a>
        <a href="#">Documentation</a>
        <a href="#">Contact</a>
    </div>
    <div class="topbar-status">
        <span class="pulse"></span>&nbsp; Système opérationnel
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-img"></div>
    <div class="hero-gradient"></div>
    <div class="hero-content">
        <div class="hero-eyebrow">
            <span class="hero-eyebrow-dot"></span>
            Dispositif de Recherche Clinique
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
                <div class="kpi-label">Gènes analysés</div>
            </div>
            <div class="kpi">
                <div class="kpi-val">500</div>
                <div class="kpi-label">Arbres décisionnels</div>
            </div>
            <div class="kpi">
                <div class="kpi-val">TCGA</div>
                <div class="kpi-label">Cohorte de référence</div>
            </div>
            <div class="kpi">
                <div class="kpi-val">57</div>
                <div class="kpi-label">Features totales</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_analyse, tab_methodo = st.tabs(["🚀  Analyse Patient", "📖  Méthodologie"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB — ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analyse:
    st.warning(
        "**Dispositif de Recherche.** Ce système génère un score de risque basé sur "
        "54 signatures transcriptomiques. Il ne remplace pas le jugement clinique d'un médecin."
    )
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([5, 7], gap="large")

    with col_in:
        st.markdown('<div class="glass"><div class="glass-title">📋 &nbsp;Paramètres Cliniques</div>', unsafe_allow_html=True)
        age   = st.number_input("Âge du patient", min_value=1, max_value=115, value=55)
        sexe  = st.radio("Sexe biologique", ["Homme", "Femme"], horizontal=True)
        stade = st.selectbox("Stade TNM initial", ["I", "II", "III", "IV"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        st.markdown('<div class="glass"><div class="glass-title">🔬 &nbsp;Données Omiques</div>', unsafe_allow_html=True)
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
                badge_txt  = "&#11044; &nbsp;Risque Faible"
                decision   = ("<strong>Recommandation</strong>"
                              "Mélanome primaire probable. Surveillance standard et "
                              "suivi dermatologique classique recommandé.")
                bar_color  = "#00c9a7"
            elif prob < 0.67:
                rc, bc, pc = "result-med",  "badge-med",  "prob-med"
                badge_txt  = "&#11044; &nbsp;Risque Intermédiaire"
                decision   = ("<strong>Recommandation</strong>"
                              "Zone d'incertitude clinique. Examens complémentaires, "
                              "confirmation histologique et suivi rapproché.")
                bar_color  = "#ffbb00"
            else:
                rc, bc, pc = "result-high", "badge-high", "prob-high"
                badge_txt  = "&#11044; &nbsp;Risque Élevé"
                decision   = ("<strong>Recommandation</strong>"
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB — MÉTHODOLOGIE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_methodo:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title">Architecture du Modèle</div>
            <ul>
                <li><strong>Signature Génomique :</strong> 54 biomarqueurs mRNA sélectionnés par régression Lasso — invasion tumorale, remodelage de la MEC, EMT, inflammation.</li>
                <li><strong>Moteur prédictif :</strong> Random Forest de 500 arbres décisionnels.</li>
                <li><strong>Cohorte :</strong> Entraîné sur TCGA-SKCM (mélanome cutané).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="mcard">
            <div class="mcard-title">Normalisation Z-score</div>
            <p>Chaque échantillon subit une normalisation basée sur les paramètres statistiques de la cohorte TCGA de référence :</p>
            <div class="formula-box">z = (x &minus; &mu;) / &sigma;</div>
            <p style="margin-top:0.9rem">&mu; et &sigma; sont calculés sur les distributions TCGA-SKCM pour chacun des 54 gènes et 3 variables cliniques.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title">Procédure Diagnostique</div>
            <div class="step-row">
                <div class="step-dot">1</div>
                <div><strong>Input :</strong> Saisie des paramètres cliniques (Âge, Sexe, Stade TNM) et chargement du profil d'expression 54 gènes au format .csv.</div>
            </div>
            <div class="step-row">
                <div class="step-dot">2</div>
                <div><strong>Fusion Multimodale :</strong> Encodage et concaténation pour former un vecteur unique de 57 variables (54G + 3C).</div>
            </div>
            <div class="step-row">
                <div class="step-dot">3</div>
                <div><strong>Standardisation :</strong> Application des moyennes et écarts-types TCGA sur chaque variable individuelle.</div>
            </div>
            <div class="step-row">
                <div class="step-dot">4</div>
                <div><strong>Prédiction :</strong> Calcul de la probabilité p métastatique via le modèle Random Forest entraîné.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="mcard">
            <div class="mcard-title">Seuils de Décision Clinique</div>
            <div class="threshold-row">
                <div class="th-ind th-low"></div>
                <div>
                    <div class="th-title">Risque Faible — p &lt; 33%</div>
                    <div class="th-desc">Mélanome primaire probable. Surveillance standard recommandée.</div>
                </div>
            </div>
            <div class="threshold-row">
                <div class="th-ind th-med"></div>
                <div>
                    <div class="th-title">Risque Intermédiaire — 33% &le; p &lt; 67%</div>
                    <div class="th-desc">Zone d'incertitude. Confirmation histologique et suivi rapproché.</div>
                </div>
            </div>
            <div class="threshold-row">
                <div class="th-ind th-high"></div>
                <div>
                    <div class="th-title">Risque Élevé — p &ge; 67%</div>
                    <div class="th-desc">Mélanome métastatique probable. Discussion précoce immunothérapie (anti-PD-1 / anti-CTLA-4).</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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

    st.markdown("**Statut Système**")
    if model_ok:
        st.success("✅  Modèle chargé")
        st.info("🌲  Random Forest · 500 arbres")
        st.info("🔬  57 features · 54G + 3C")
    else:
        st.error("❌  Modèle introuvable")
        st.warning("Vérifiez `model_multimodal_54.pkl` et `params_multimodal_54.json`")

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem; color:rgba(0,0,0,0.35);
                line-height:1.7; padding:0 0.2rem;">
        Cet outil est réservé à un usage de recherche. Il ne constitue pas
        un dispositif médical certifié et ne remplace pas l'avis d'un
        professionnel de santé qualifié.
    </div>
    """, unsafe_allow_html=True)
