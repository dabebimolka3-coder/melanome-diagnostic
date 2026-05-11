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
if 'current_page' not in st.session_state:
    page_param = st.query_params.get("page", "analyse")
    st.session_state['current_page'] = page_param

def navigate_to(page):
    st.session_state['current_page'] = page
    st.query_params["page"] = page
    st.rerun()

# ── GLOBAL CSS AVEC IMAGE D'ARRIÈRE-PLAN POUR TOUTE LA PAGE ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Syne:wght@400;500;600;700&display=swap');

/* Image d'arrière-plan pour TOUTE l'application */
.stApp {
    background-image: url('https://images.squarespace-cdn.com/content/v1/5d9e30182db9d71681f4a692/1581717140307-89XZXBK2C5OBW2AGDXCO/mountainviewrheumatoidarthritis.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}

/* Overlay semi-transparent pour améliorer la lisibilité */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.92);
    z-index: 0;
    pointer-events: none;
}

/* Assure que tout le contenu est au-dessus de l'overlay */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    position: relative;
    z-index: 1;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

[data-testid="block-container"] {
    padding: 0 2rem 3rem !important;
    max-width: 1400px;
    margin: 0 auto;
    background: transparent !important;
}

* {
    font-family: 'Syne', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #0d1b2a !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none;
}

/* Topbar avec fond légèrement transparent */
.topbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(0,0,0,0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: 60px;
    margin: 0 -2rem 2rem;
}

.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.topbar-logo {
    width: 34px;
    height: 34px;
    background: #0d1b2a;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: white;
}

.topbar-name {
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem;
    font-weight: 600;
    color: #0d1b2a;
}

.topbar-name span {
    color: #1a6fff;
}

.topbar-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    font-weight: 500;
    color: #2c7a5e;
    background: rgba(230, 247, 240, 0.9);
    padding: 4px 12px;
    border-radius: 20px;
}

.pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00c9a7;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.2); }
}

/* Hero section avec fond transparent */
.hero-section {
    margin-bottom: 2rem;
    position: relative;
}

.hero-content {
    padding: 2rem 0 1rem;
    max-width: 800px;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(26,111,255,0.1);
    backdrop-filter: blur(5px);
    border: 1px solid rgba(26,111,255,0.25);
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.68rem;
    color: #1a6fff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 1.1rem;
}

.hero-eyebrow-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #1a6fff;
    display: inline-block;
}

.hero h1 {
    font-size: 3rem !important;
    font-weight: 600 !important;
    color: #0d1b2a !important;
    line-height: 1.1 !important;
    margin: 0 0 1rem !important;
    letter-spacing: -0.01em;
}

.hero h1 span {
    color: #1a6fff;
}

.hero-sub {
    color: rgba(0,0,0,0.6);
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 520px;
}

.hero-kpis {
    display: flex;
    gap: 0;
    margin-top: 2rem;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    overflow: hidden;
    width: fit-content;
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(5px);
}

.kpi {
    padding: 0.8rem 1.5rem;
    border-right: 1px solid rgba(0,0,0,0.08);
}

.kpi:last-child {
    border-right: none;
}

.kpi-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0d1b2a;
    line-height: 1;
}

.kpi-label {
    font-size: 0.65rem;
    color: rgba(0,0,0,0.5);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* Cartes avec fond semi-transparent */
.glass, .mcard, .result-card {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.glass-title, .mcard-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #0d1b2a;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}

/* Inputs avec fond adapté */
.stNumberInput input, .stSelectbox select {
    background: rgba(245,245,245,0.9) !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stRadio"] label {
    background: rgba(245,245,245,0.9) !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
}

[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(26,111,255,0.15) !important;
    border-color: #1a6fff !important;
    color: #1a6fff !important;
}

/* Boutons */
.stButton > button {
    background: linear-gradient(135deg, #1a6fff, #0052d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 1.8rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    transition: all .25s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(26,111,255,0.4) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(250,250,250,0.9) !important;
    border: 2px dashed rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
}

/* Result cards spécifiques */
.result-low {
    background: rgba(0,201,167,0.12) !important;
    border-color: rgba(0,201,167,0.3) !important;
}
.result-med {
    background: rgba(255,187,0,0.12) !important;
    border-color: rgba(255,187,0,0.3) !important;
}
.result-high {
    background: rgba(255,75,75,0.12) !important;
    border-color: rgba(255,75,75,0.3) !important;
}

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.badge-low {
    background: rgba(0,201,167,0.2) !important;
    color: #00a87e;
}
.badge-med {
    background: rgba(255,187,0,0.2) !important;
    color: #cc9500;
}
.badge-high {
    background: rgba(255,75,75,0.2) !important;
    color: #cc3b3b;
}

.result-prob {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.prob-low { color: #00a87e; }
.prob-med { color: #cc9500; }
.prob-high { color: #cc3b3b; }

.result-decision {
    background: rgba(0,0,0,0.03);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
}

/* Placeholder */
.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 350px;
    background: rgba(250,250,250,0.9) !important;
    border: 1px dashed rgba(0,0,0,0.12);
    border-radius: 16px;
    text-align: center;
    gap: 0.8rem;
}

/* Step et formulaire */
.step-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1rem;
    font-size: 0.8rem;
}

.step-dot {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(26,111,255,0.15);
    border: 1px solid rgba(26,111,255,0.4);
    color: #1a6fff;
    font-size: 0.65rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}

.formula-box {
    background: rgba(26,111,255,0.1);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    color: #1a6fff;
}

.threshold-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 0.8rem;
}

.th-ind {
    width: 4px;
    height: 36px;
    border-radius: 2px;
}
.th-low { background: #00c9a7; }
.th-med { background: #ffbb00; }
.th-high { background: #ff4b4b; }

/* Sidebar avec fond semi-transparent */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(0,0,0,0.08);
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

.stProgress > div > div > div {
    border-radius: 50px !important;
    height: 4px !important;
    background: rgba(0,0,0,0.05) !important;
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

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🧬</div>
        <span class="topbar-name">MelanomaPredict <span>AI</span></span>
    </div>
    <div class="topbar-status">
        <span class="pulse"></span>
        <span>Operational</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── BOUTONS DE NAVIGATION ─────────────────────────────────────────────────────
nav_cols = st.columns(4)
nav_pages = ["analyse", "methodologie", "documentation", "contact"]
nav_labels = ["🔬 ANALYSE", "📊 MÉTHODOLOGIE", "📚 DOCUMENTATION", "📧 CONTACT"]
nav_icons = ["🔬", "📊", "📚", "📧"]

for i, (col, page, label, icon) in enumerate(zip(nav_cols, nav_pages, nav_labels, nav_icons)):
    with col:
        is_active = st.session_state['current_page'] == page
        button_label = f"{icon} {label.split()[1]}" if is_active else label
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

# PAGE MÉTHODOLOGIE
elif st.session_state['current_page'] == 'methodologie':
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">Méthodologie</h1>
        <p style="color: rgba(0,0,0,0.55); font-size: 0.9rem;">Architecture et validation du modèle prédictif</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Architecture du Modèle</div>
            <ul>
                <li><strong>Signature Génomique :</strong> 54 biomarqueurs mRNA sélectionnés par régression Lasso — invasion tumorale, remodelage de la MEC, EMT, inflammation.</li>
                <li><strong>Moteur prédictif :</strong> Random Forest de 500 arbres décisionnels.</li>
                <li><strong>Cohorte :</strong> Entraîné sur TCGA-SKCM (mélanome cutané).</li>
                <li><strong>Validation croisée :</strong> 10 folds stratifiés</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Normalisation Z-score</div>
            <p>Chaque échantillon subit une normalisation basée sur les paramètres statistiques de la cohorte TCGA de référence :</p>
            <div class="formula-box">z = (x &minus; &mu;) / &sigma;</div>
            <p style="margin-top:1rem">&mu; et &sigma; sont calculés sur les distributions TCGA-SKCM pour chacun des 54 gènes et 3 variables cliniques.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Procédure Diagnostique</div>
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

        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Seuils de Décision Clinique</div>
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

# PAGE DOCUMENTATION
elif st.session_state['current_page'] == 'documentation':
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">Documentation Scientifique</h1>
        <p style="color: rgba(0,0,0,0.55); font-size: 0.9rem;">Modèle multimodal pour la prédiction du risque métastatique dans le mélanome cutané</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Base de Données</div>
            <ul>
                <li><strong>Cohorte :</strong> TCGA-SKCM (Skin Cutaneous Melanoma)</li>
                <li><strong>Échantillons :</strong> 473 patients (mélanome primaire et métastatique)</li>
                <li><strong>Features :</strong> 57 variables (54 gènes + 3 cliniques)</li>
                <li><strong>Ratio :</strong> Train/Test 80/20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    
    with col2:
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Performances</div>
            <ul>
                <li><strong>Accuracy :</strong> 90%</li>
                <li><strong>Sensibilité :</strong> 85%</li>
                <li><strong>Spécificité :</strong> 95%</li>
                <li><strong>AUC-ROC :</strong> 0.955</li>
                <li><strong>F1-Score :</strong> 89.47%</li>
                <li><strong>Precision :</strong> 94.44%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mcard">
            <div class="mcard-title"> Limitations</div>
            <ul>
                <li>Validation externe en cours sur cohorte indépendante</li>
                <li>Ne remplace pas le gold standard histologique</li>
                <li>Usage réservé à la recherche clinique</li>
                <li>Nécessite normalisation standardisée des expressions géniques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    

# PAGE CONTACT
elif st.session_state['current_page'] == 'contact':
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">Contact & Support</h1>
        <p style="color: rgba(0,0,0,0.55); font-size: 0.9rem;">Une question, une collaboration ou un support technique ? N'hésitez pas à nous contacter.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mcard">
        <div class="mcard-title"> Formulaire de contact</div>
        <p>Pour toute demande d'information ou collaboration, veuillez nous envoyer un email à :</p>
        <p style="font-family: monospace; font-size: 1rem; color: #1a6fff; text-align: center; margin-top: 0.5rem;">contact@melanomapredict.ai</p>
    </div>
    """, unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 1.8rem;">🧬</div>
        <div style="font-weight: 600;">MelanomaPredict AI</div>
        <div style="font-size: 0.6rem; opacity: 0.5;">v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔬 Analyse", use_container_width=True):
        navigate_to('analyse')
    if st.button("📊 Méthodologie", use_container_width=True):
        navigate_to('methodologie')
    if st.button("📚 Documentation", use_container_width=True):
        navigate_to('documentation')
    if st.button("📧 Contact", use_container_width=True):
        navigate_to('contact')
    
    st.divider()
    
    if model_ok:
        st.success("✅ Modèle chargé")
        st.caption("🌲 Random Forest · 57 features")
    else:
        st.error("❌ Modèle introuvable")
    
    st.divider()
    st.caption("⚠️ Usage recherche uniquement")
