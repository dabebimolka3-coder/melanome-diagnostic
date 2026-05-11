import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MelanomaPredict AI | Portail Diagnostic",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE CSS PERSONNALISÉ ---
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@600;700&display=swap');

.main { background-color: #f4f7fc; }

[data-testid="stAppViewContainer"] {
    background-color: #f4f7fc;
}

.main-header {
    background: linear-gradient(90deg, #002b5c 0%, #004aad 100%);
    padding: 2.5rem 3rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(0,43,92,0.18);
    display: flex;
    align-items: center;
    gap: 2rem;
}

.header-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    color: #ffffff;
}

.header-text p {
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
    margin: 0;
}

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.header-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
}

.stat-item strong {
    display: block;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00b4d8;
}

.stat-item small {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 0.04em;
}

.report-card {
    background-color: white;
    padding: 2rem;
    border-radius: 14px;
    border-left: 6px solid #002b5c;
    box-shadow: 0 4px 20px rgba(0,43,92,0.10);
    margin-bottom: 1.5rem;
}

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #002b5c;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #dde4f0;
}

.risk-low  { border-left-color: #2d6a4f !important; }
.risk-med  { border-left-color: #e9a700 !important; }
.risk-high { border-left-color: #d62828 !important; }

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.badge-low  { background: #d4edda; color: #155724; }
.badge-med  { background: #fff3cd; color: #856404; }
.badge-high { background: #f8d7da; color: #721c24; }

.prob-display {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #002b5c;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.decision-box {
    background: #f4f7fc;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.875rem;
    line-height: 1.6;
    color: #1a1a2e;
    border: 1px solid #dde4f0;
    margin-top: 1rem;
}

.decision-box strong {
    color: #002b5c;
    display: block;
    margin-bottom: 4px;
    font-weight: 600;
}

.method-card {
    background: white;
    border: 1px solid #dde4f0;
    border-radius: 14px;
    padding: 1.6rem;
    box-shadow: 0 4px 20px rgba(0,43,92,0.08);
    margin-bottom: 1.2rem;
}

.step-item {
    display: flex;
    gap: 12px;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    color: #6b7280;
    line-height: 1.6;
}

.step-num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #002b5c;
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.stButton > button {
    border-radius: 50px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

.stDownloadButton > button {
    border-radius: 50px !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] {
    background: #002b5c !important;
}

[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}

[data-testid="stSidebar"] .stSuccess > div {
    background: rgba(82,183,136,0.2) !important;
    border: 1px solid rgba(82,183,136,0.4) !important;
}

[data-testid="stSidebar"] .stInfo > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

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
    except Exception:
        return None, None

model, params = load_assets()

# --- 4. HEADER PRINCIPAL ---
header_html = """
<div class="main-header">
    <div class="header-text">
        <div class="header-badge">&#9679; Dispositif de Recherche Clinique</div>
        <h1>MelanomaPredict AI &#129
742;</h1>
        <p>Aide &agrave; la d&eacute;cision th&eacute;rapeutique par analyse de 54 biomarqueurs transcriptomiques</p>
        <div class="header-stats">
            <div class="stat-item"><strong>54</strong><small>G&egrave;nes analys&eacute;s</small></div>
            <div class="stat-item"><strong>TCGA-SKCM</strong><small>Cohorte de r&eacute;f&eacute;rence</small></div>
            <div class="stat-item"><strong>Random Forest</strong><small>500 arbres d&eacute;cisionnels</small></div>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# --- 5. NAVIGATION ---
tab1, tab2 = st.tabs(["🚀 Analyse Patient", "📖 Méthodologie"])

# --- MÉTHODOLOGIE ---
with tab2:
    st.markdown("## Méthodologie Scientifique & Technique")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="method-card">
            <div class="card-title">1. Architecture du Modèle</div>
            <ul style="font-size:0.875rem; color:#6b7280; line-height:1.7; padding-left:1.2rem;">
                <li><strong style="color:#002b5c;">Signature Génomique :</strong> 54 biomarqueurs (mRNA) sélectionnés par régression Lasso — invasion, Remodelage MEC, EMT, Inflammation.</li>
                <li style="margin-top:0.5rem"><strong style="color:#002b5c;">Moteur :</strong> Random Forest de 500 arbres décisionnels.</li>
                <li style="margin-top:0.5rem"><strong style="color:#002b5c;">Source :</strong> Cohorte TCGA-SKCM.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <div class="card-title">2. Normalisation Z-score</div>
            <p style="font-size:0.875rem; color:#6b7280; line-height:1.7;">
                Chaque échantillon subit une normalisation Z-score basée sur les paramètres de la cohorte de référence :
            </p>
            <div style="background:#002b5c; color:#fff; font-family:monospace; font-size:1.2rem; text-align:center; padding:1rem; border-radius:10px; margin-top:0.8rem; letter-spacing:0.05em;">
                z = (x &minus; &mu;) / &sigma;
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="method-card">
            <div class="card-title">3. Procédure Diagnostique</div>
            <div class="step-item">
                <div class="step-num">1</div>
                <div><strong style="color:#002b5c;">Input :</strong> Saisie des paramètres cliniques et chargement du profil d'expression 54 gènes (.csv).</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div><strong style="color:#002b5c;">Fusion Multimodale :</strong> Encodage et concaténation pour former un profil unique de 57 variables.</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div><strong style="color:#002b5c;">Standardisation :</strong> Application des moyennes et écarts-types TCGA.</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div><strong style="color:#002b5c;">Prédiction :</strong> Calcul de la probabilité p métastatique via le modèle Random Forest.</div>
            </div>
        </div>

        <div class="method-card">
            <div class="card-title">4. Seuils de Décision</div>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:12px;height:12px;border-radius:50%;background:#2d6a4f;flex-shrink:0;"></div>
                    <div><strong style="color:#2d6a4f;">Risque Faible (p &lt; 33%)</strong><br/><span style="font-size:0.82rem;color:#6b7280;">Surveillance standard.</span></div>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:12px;height:12px;border-radius:50%;background:#e9a700;flex-shrink:0;"></div>
                    <div><strong style="color:#856404;">Risque Intermédiaire (33–67%)</strong><br/><span style="font-size:0.82rem;color:#6b7280;">Confirmation histologique et suivi rapproché.</span></div>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:12px;height:12px;border-radius:50%;background:#d62828;flex-shrink:0;"></div>
                    <div><strong style="color:#d62828;">Risque Élevé (p &gt; 67%)</strong><br/><span style="font-size:0.82rem;color:#6b7280;">Discussion précoce immunothérapie (anti-PD-1 / anti-CTLA-4).</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- ANALYSE PATIENT ---
with tab1:
    st.warning("**Dispositif de Recherche** : Score de risque basé sur 54 signatures transcriptomiques. Ne remplace pas l'avis clinique.")

    col_input, col_display = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown('<div class="card-title">📋 Paramètres Cliniques</div>', unsafe_allow_html=True)
        age = st.number_input("Âge du patient", 1, 115, 55)
        sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        stade = st.selectbox("Stade Initial", ["I", "II", "III", "IV"])

        st.divider()
        st.markdown('<div class="card-title">📥 Données Omiques</div>', unsafe_allow_html=True)

        if params:
            example_df = pd.DataFrame(
                np.random.uniform(0.5, 5.0, size=(1, 54)),
                columns=params['top_genes']
            )
            csv_example = example_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger le Template (.csv)",
                data=csv_example,
                file_name="template_54_genes.csv",
                mime="text/csv",
                use_container_width=True
            )

        uploaded_file = st.file_uploader(
            "Charger le profil d'expression (HGNC Symbols)",
            type="csv"
        )

        if uploaded_file and model and params:
            df_patient = pd.read_csv(uploaded_file)
            if st.button("🚀 Lancer le Diagnostic", use_container_width=True, type="primary"):
                with st.spinner("Analyse du profil en cours…"):
                    sexe_val = 0 if sexe == "Homme" else 1
                    stade_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
                    stade_val = stade_map[stade]

                    clinique_vec = [age, sexe_val, stade_val]
                    omique_vec = df_patient[params['top_genes']].iloc[0].tolist()
                    X_combined = np.array(clinique_vec + omique_vec).reshape(1, -1)

                    X_scaled = (X_combined - np.array(params['means'])) / np.array(params['stds'])
                    prob = model.predict_proba(X_scaled)[0][1]

                    st.session_state['analysis'] = {
                        'prob': prob,
                        'top_genes': params['top_genes']
                    }

    with col_display:
        if 'analysis' in st.session_state:
            res = st.session_state['analysis']
            prob_val = res['prob']
            pct = prob_val * 100

            # Determine risk level
            if prob_val < 0.33:
                risk_class = "risk-low"
                badge_class = "badge-low"
                badge_text = "🟢 RISQUE FAIBLE"
                decision_text = "<strong>Décision</strong>Mélanome primaire probable. Surveillance standard et suivi dermatologique classique recommandé."
            elif prob_val < 0.67:
                risk_class = "risk-med"
                badge_class = "badge-med"
                badge_text = "🟡 RISQUE INTERMÉDIAIRE"
                decision_text = "<strong>Décision</strong>Zone d'incertitude clinique. Examens complémentaires, confirmation histologique et suivi rapproché."
            else:
                risk_class = "risk-high"
                badge_class = "badge-high"
                badge_text = "🔴 RISQUE ÉLEVÉ"
                decision_text = "<strong>Décision</strong>Mélanome métastatique probable. Discussion précoce d'immunothérapie (anti-PD-1 / anti-CTLA-4) ou thérapie ciblée."

            report_html = f"""
            <div class="report-card {risk_class}">
                <span class="badge {badge_class}">{badge_text}</span>
                <div class="prob-display">{pct:.1f}%</div>
                <p style="color:#6b7280; font-size:0.82rem; margin-bottom:0.8rem;">Probabilité Métastatique (Score p)</p>
                <div class="decision-box">{decision_text}</div>
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)
            st.progress(prob_val)

            # Biomarker chart
            importances = model.feature_importances_[3:]
            top_10_df = (
                pd.DataFrame({'Gène': res['top_genes'], 'Importance': importances})
                .sort_values('Importance', ascending=False)
                .head(10)
            )

            fig = px.bar(
                top_10_df,
                x='Importance',
                y='Gène',
                orientation='h',
                color='Importance',
                title="Top 10 des Biomarqueurs Décisifs",
                color_continuous_scale=[[0, '#e8f0fb'], [1, '#002b5c']],
                template="simple_white"
            )
            fig.update_layout(
                title_font_family="Georgia",
                title_font_size=16,
                title_font_color="#002b5c",
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 2rem; background:white; border-radius:14px; border:1px solid #dde4f0;">
                <div style="font-size:3rem; margin-bottom:1rem;">🧬</div>
                <p style="color:#6b7280; font-size:0.9rem; line-height:1.6;">
                    Chargez un profil d'expression génique<br/>puis lancez le diagnostic pour voir les résultats.
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🧬</div>
        <div style="font-family:Georgia,serif; font-size:1rem; font-weight:700; color:white; margin-bottom:0.3rem;">
            MelanomaPredict AI
        </div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.5); letter-spacing:0.08em; text-transform:uppercase;">
            Portail Diagnostic
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Statut Système**")

    if model:
        st.success("✅ Modèle : Opérationnel")
        st.info("🌲 Algorithme : Random Forest")
        st.info("🔬 Features : 54 Gènes + 3 Cliniques")
    else:
        st.error("❌ Ressources manquantes")
        st.warning("Vérifiez model_multimodal_54.pkl et params_multimodal_54.json")
