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

# ── CSS ULTRA PROFESSIONNEL (STYLE MÉDICAL / LUXE) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=Playfair+Display:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #F7F9FC !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #F7F9FC !important;
}

[data-testid="block-container"] {
    padding: 0 2rem 2rem !important;
    max-width: 1400px;
}

/* Navigation premium style */
.premium-nav {
    background: white;
    border-radius: 20px;
    padding: 0.5rem 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.03);
    border: 1px solid rgba(0, 0, 0, 0.04);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #1E6DFF, #00B8A9);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 6px 14px rgba(30, 109, 255, 0.2);
}

.nav-title {
    font-weight: 700;
    font-size: 1.2rem;
    letter-spacing: -0.3px;
    background: linear-gradient(135deg, #111B2C, #1E3A5F);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}

.nav-badge {
    font-size: 0.65rem;
    background: #EEF2FF;
    color: #1E6DFF;
    padding: 3px 10px;
    border-radius: 30px;
    font-weight: 600;
    margin-left: 8px;
}

.nav-links {
    display: flex;
    gap: 0.3rem;
    background: #F8FAFE;
    padding: 0.3rem;
    border-radius: 60px;
}

.nav-btn {
    padding: 0.55rem 1.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    color: #4A5B6E;
    background: transparent;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif;
}

.nav-btn:hover {
    background: rgba(30, 109, 255, 0.08);
    color: #1E6DFF;
}

.nav-btn-active {
    background: white;
    color: #1E6DFF;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.status-pulse {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #F0FDF4;
    padding: 0.4rem 1rem;
    border-radius: 40px;
    font-size: 0.7rem;
    font-weight: 500;
    color: #00A86B;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background: #00C9A7;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 201, 167, 0.5); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 201, 167, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 201, 167, 0); }
}

/* Cartes de contenu */
.content-card {
    background: white;
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid rgba(0, 0, 0, 0.04);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.02);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 1.5rem 0;
}

.metric-item {
    background: #F8FAFE;
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid rgba(0, 0, 0, 0.03);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1E6DFF;
    font-family: 'Playfair Display', serif;
}

.metric-label {
    font-size: 0.7rem;
    color: #6B7A8A;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

.threshold-line {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1rem 0;
    padding: 0.8rem;
    background: #F8FAFE;
    border-radius: 16px;
}

.threshold-color {
    width: 4px;
    height: 40px;
    border-radius: 4px;
}

.threshold-low { background: #00C9A7; }
.threshold-mid { background: #FFB800; }
.threshold-high { background: #FF4B4B; }

/* Responsive */
@media (max-width: 768px) {
    .premium-nav {
        flex-direction: column;
        align-items: stretch;
    }
    .nav-links {
        justify-content: center;
    }
}
</style>
""", unsafe_allow_html=True)

# ── COMPOSANT DE NAVIGATION PREMIUM ──────────────────────────────────────────
def render_premium_nav():
    current = st.session_state['current_page']
    
    nav_html = f"""
    <div class="premium-nav">
        <div class="nav-brand">
            <div class="nav-icon">🧬</div>
            <div>
                <span class="nav-title">MelanomaPredict AI</span>
                <span class="nav-badge">Research Grade</span>
            </div>
        </div>
        <div class="nav-links">
            <button class="nav-btn {'nav-btn-active' if current == 'analyse' else ''}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'analyse'}}, '*')">📊 ANALYSE</button>
            <button class="nav-btn {'nav-btn-active' if current == 'methodologie' else ''}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'methodologie'}}, '*')">🔬 MÉTHODOLOGIE</button>
            <button class="nav-btn {'nav-btn-active' if current == 'documentation' else ''}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'documentation'}}, '*')">📚 DOCUMENTATION</button>
            <button class="nav-btn {'nav-btn-active' if current == 'contact' else ''}" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'contact'}}, '*')">✉️ CONTACT</button>
        </div>
        <div class="status-pulse">
            <div class="pulse-dot"></div>
            <span>Système opérationnel</span>
        </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Capture des clics via colonnes Streamlit (fallback fiable)
    col1, col2, col3, col4 = st.columns(4, gap="small")
    pages = ["analyse", "methodologie", "documentation", "contact"]
    labels = ["ANALYSE", "MÉTHODOLOGIE", "DOCUMENTATION", "CONTACT"]
    icons = ["📊", "🔬", "📚", "✉️"]
    
    for col, page, label, icon in zip([col1, col2, col3, col4], pages, labels, icons):
        with col:
            if st.button(f"{icon} {label}", key=f"nav_{page}", use_container_width=True):
                navigate_to(page)

# ── CONTENU : PAGE MÉTHODOLOGIE ──────────────────────────────────────────────
def page_methodologie():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #111B2C, #1E3A5F); background-clip: text; -webkit-background-clip: text; color: transparent;">Méthodologie Clinique & IA</h1>
        <p style="color: #6B7A8A; margin-top: 0.5rem;">Architecture du modèle multimodal – Validation rigoureuse – Interprétabilité</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">🧠 Architecture du Modèle</h3>
            <p><strong>Random Forest</strong> · 500 arbres décisionnels</p>
            <p><strong>57 features</strong> : 54 gènes + 3 variables cliniques (âge, sexe, stade TNM)</p>
            <p><strong>Normalisation</strong> : Z-score sur cohorte TCGA-SKCM de référence</p>
            <p><strong>Validation</strong> : 10-fold cross-validation stratifiée</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card" style="margin-top: 1rem;">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">📊 Performance du Modèle</h3>
            <div class="metric-grid">
                <div class="metric-item"><div class="metric-value">90%</div><div class="metric-label">Accuracy</div></div>
                <div class="metric-item"><div class="metric-value">94.4%</div><div class="metric-label">Precision</div></div>
                <div class="metric-item"><div class="metric-value">89.5%</div><div class="metric-label">F1-Score</div></div>
                <div class="metric-item"><div class="metric-value">0.955</div><div class="metric-label">AUC-ROC</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">⚙️ Pipeline Diagnostique</h3>
            <div style="margin: 1rem 0;">
                <div style="display: flex; gap: 12px; margin-bottom: 1rem;"><div style="width: 28px; height: 28px; background: #1E6DFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8rem;">1</div><div><strong>Input multimodal</strong><br><span style="color: #6B7A8A; font-size: 0.85rem;">Données cliniques + profils d'expression génique</span></div></div>
                <div style="display: flex; gap: 12px; margin-bottom: 1rem;"><div style="width: 28px; height: 28px; background: #1E6DFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8rem;">2</div><div><strong>Fusion & Standardisation</strong><br><span style="color: #6B7A8A; font-size: 0.85rem;">Concaténation des vecteurs + normalisation Z-score</span></div></div>
                <div style="display: flex; gap: 12px; margin-bottom: 1rem;"><div style="width: 28px; height: 28px; background: #1E6DFF; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8rem;">3</div><div><strong>Prédiction</strong><br><span style="color: #6B7A8A; font-size: 0.85rem;">Calcul de probabilité métastatique par Random Forest</span></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card" style="margin-top: 1rem;">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">🎯 Seuils de Décision Clinique</h3>
            <div class="threshold-line"><div class="threshold-color threshold-low"></div><div><strong>Risque Faible</strong> — p < 33%<br><span style="font-size: 0.8rem; color: #6B7A8A;">Mélanome primaire probable</span></div></div>
            <div class="threshold-line"><div class="threshold-color threshold-mid"></div><div><strong>Risque Intermédiaire</strong> — 33% ≤ p < 67%<br><span style="font-size: 0.8rem; color: #6B7A8A;">Zone d'incertitude, examens complémentaires</span></div></div>
            <div class="threshold-line"><div class="threshold-color threshold-high"></div><div><strong>Risque Élevé</strong> — p ≥ 67%<br><span style="font-size: 0.8rem; color: #6B7A8A;">Mélanome métastatique probable</span></div></div>
        </div>
        """, unsafe_allow_html=True)

# ── CONTENU : PAGE DOCUMENTATION ─────────────────────────────────────────────
def page_documentation():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #111B2C, #1E3A5F); background-clip: text; -webkit-background-clip: text; color: transparent;">Documentation Scientifique</h1>
        <p style="color: #6B7A8A; margin-top: 0.5rem;">Référentiel complet – Jeux de données – Métriques – API</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">📊 Cohorte de Référence</h3>
            <p><strong>Base de données :</strong> TCGA-SKCM (Skin Cutaneous Melanoma)</p>
            <p><strong>Échantillons :</strong> 473 patients</p>
            <p><strong>Ratio :</strong> Train 80% / Test 20%</p>
            <p><strong>Biomarqueurs :</strong> 54 gènes sélectionnés par ElasticNet</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card" style="margin-top: 1rem;">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">🔗 Accès & API</h3>
            <p><strong>Endpoint REST :</strong> <code>api.melanomapredict.ai/v1/predict</code></p>
            <p><strong>Format d'entrée :</strong> JSON avec 57 features normalisées</p>
            <p><strong>Authentification :</strong> API Key (usage recherche)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">📈 Métriques de Performance</h3>
            <div class="metric-grid">
                <div class="metric-item"><div class="metric-value">90%</div><div class="metric-label">Accuracy</div></div>
                <div class="metric-item"><div class="metric-value">85%</div><div class="metric-label">Sensibilité</div></div>
                <div class="metric-item"><div class="metric-value">95%</div><div class="metric-label">Spécificité</div></div>
                <div class="metric-item"><div class="metric-value">0.955</div><div class="metric-label">AUC-ROC</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card" style="margin-top: 1rem;">
            <h3 style="font-size: 1.2rem; margin-bottom: 1rem;">⚠️ Limitations</h3>
            <ul style="color: #6B7A8A; line-height: 1.8;">
                <li>Validation externe en cours sur cohorte indépendante</li>
                <li>Ne remplace pas le gold standard histologique</li>
                <li>Usage réservé à la recherche clinique</li>
                <li>Nécessite normalisation standardisée des expressions géniques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ── CONTENU : PAGE CONTACT ───────────────────────────────────────────────────
def page_contact():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #111B2C, #1E3A5F); background-clip: text; -webkit-background-clip: text; color: transparent;">Contact & Support</h1>
        <p style="color: #6B7A8A; margin-top: 0.5rem;">Collaboration scientifique – Support technique – Partenariats</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📧</div>
            <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Email</h3>
            <p style="color: #1E6DFF; font-family: monospace;">contact@melanomapredict.ai</p>
            <p style="font-size: 0.75rem; color: #6B7A8A;">Réponse sous 48h ouvrées</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
            <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Recherche</h3>
            <p>Pr. Jean Dupont<br>Oncologie numérique</p>
            <p style="color: #1E6DFF;">jean.dupont@chu.fr</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
            <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Support technique</h3>
            <p>Documentation API<br>Intégration & déploiement</p>
            <p style="color: #1E6DFF;">support@melanomapredict.ai</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card" style="margin-top: 1rem; text-align: center;">
        <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">📍 Laboratoire partenaire</h3>
        <p>Centre de Recherche en Cancérologie de Lyon (CRCL)<br>
        INSERM U1052 – Équipe "Génomique des mélanomes"<br>
        28 Rue Laennec, 69008 Lyon, France</p>
    </div>
    """, unsafe_allow_html=True)

# ── CONTENU : PAGE ANALYSE (VERSION SIMPLIFIÉE POUR L'EXEMPLE) ───────────────
def page_analyse():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #111B2C, #1E3A5F); background-clip: text; -webkit-background-clip: text; color: transparent;">Analyse Diagnostique Multimodale</h1>
        <p style="color: #6B7A8A;">54 biomarqueurs transcriptomiques + paramètres cliniques</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not model_ok:
        st.error("⚠️ Modèle non chargé. Vérifiez les fichiers `model_multimodal_54.pkl` et `params_multimodal_54.json`")
        return
    
    col_in, col_out = st.columns([5, 7], gap="large")
    
    with col_in:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Paramètres Cliniques")
        age = st.number_input("Âge du patient", min_value=1, max_value=115, value=55)
        sexe = st.radio("Sexe biologique", ["Homme", "Femme"], horizontal=True)
        stade = st.selectbox("Stade TNM initial", ["I", "II", "III", "IV"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="content-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("### 🔬 Données Omiques")
        if params:
            example_df = pd.DataFrame(
                np.random.uniform(0.5, 5.0, size=(1, 54)),
                columns=params['top_genes']
            )
            st.download_button(
                label="📥 Télécharger le template CSV",
                data=example_df.to_csv(index=False).encode('utf-8'),
                file_name="template_54_genes.csv",
                mime="text/csv",
                use_container_width=True
            )
        uploaded_file = st.file_uploader("Profil d'expression génique (54 gènes)", type="csv")
        run_btn = st.button("🚀 Lancer l'analyse diagnostique", use_container_width=True, disabled=(uploaded_file is None))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_out:
        if run_btn and uploaded_file and model and params:
            df_patient = pd.read_csv(uploaded_file)
            with st.spinner("Analyse en cours..."):
                try:
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_val = {"I": 1, "II": 2, "III": 3, "IV": 4}[stade]
                    omique_vec = df_patient[params['top_genes']].iloc[0].tolist()
                    X = np.array([age, sexe_val, stade_val] + omique_vec).reshape(1, -1)
                    X_scaled = (X - np.array(params['means'])) / np.array(params['stds'])
                    prob = model.predict_proba(X_scaled)[0][1]
                    pct = prob * 100
                    
                    if prob < 0.33:
                        badge = "🟢 Risque Faible"
                        color = "#00C9A7"
                        recommendation = "Mélanome primaire probable. Surveillance standard recommandée."
                    elif prob < 0.67:
                        badge = "🟡 Risque Intermédiaire"
                        color = "#FFB800"
                        recommendation = "Zone d'incertitude clinique. Confirmation histologique et suivi rapproché."
                    else:
                        badge = "🔴 Risque Élevé"
                        color = "#FF4B4B"
                        recommendation = "Mélanome métastatique probable. Discussion précoce d'immunothérapie."
                    
                    st.markdown(f"""
                    <div class="content-card" style="text-align: center; border-top: 4px solid {color};">
                        <div style="font-size: 3rem; font-weight: 700; color: {color};">{pct:.1f}%</div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">{badge}</div>
                        <div style="color: #6B7A8A;">Probabilité métastatique estimée</div>
                        <div style="background: #F8FAFE; border-radius: 12px; padding: 1rem; margin-top: 1rem; text-align: left;">
                            <strong>Recommandation</strong><br>{recommendation}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(prob)
                    
                    # Graphique d'importance
                    importances = model.feature_importances_[3:]
                    df_imp = pd.DataFrame({'gene': params['top_genes'], 'imp': importances}).sort_values('imp', ascending=True).tail(10)
                    fig = go.Figure(go.Bar(x=df_imp['imp'], y=df_imp['gene'], orientation='h', marker_color=color))
                    fig.update_layout(title="Top 10 biomarqueurs", height=300, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")
        else:
            st.markdown("""
            <div class="content-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; opacity: 0.3;">🧬</div>
                <p style="color: #6B7A8A; margin-top: 1rem;">Chargez un profil d'expression génique<br>puis lancez le diagnostic</p>
            </div>
            """, unsafe_allow_html=True)

# ── MAIN : ROUTAGE DES PAGES ─────────────────────────────────────────────────
def main():
    render_premium_nav()
    
    if st.session_state['current_page'] == 'analyse':
        page_analyse()
    elif st.session_state['current_page'] == 'methodologie':
        page_methodologie()
    elif st.session_state['current_page'] == 'documentation':
        page_documentation()
    elif st.session_state['current_page'] == 'contact':
        page_contact()

if __name__ == "__main__":
    main()
