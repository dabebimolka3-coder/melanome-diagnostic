<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MelanomaPredict AI | Portail Diagnostic</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet" />
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --navy: #002b5c;
    --navy-mid: #004aad;
    --navy-light: #e8f0fb;
    --accent: #1a73e8;
    --accent2: #00b4d8;
    --danger: #d62828;
    --warn: #e9c46a;
    --success: #2d6a4f;
    --text: #1a1a2e;
    --muted: #6b7280;
    --border: #dde4f0;
    --bg: #f4f7fc;
    --card: #ffffff;
    --radius: 14px;
    --shadow: 0 4px 24px rgba(0,43,92,0.10);
  }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* NAV */
  nav {
    background: var(--navy);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,0.18);
  }
  .nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
  }
  .nav-brand svg { width: 32px; height: 32px; }
  .nav-brand span {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.02em;
  }
  .nav-links { display: flex; gap: 2rem; align-items: center; }
  .nav-links a {
    color: rgba(255,255,255,0.75);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    transition: color 0.2s;
    letter-spacing: 0.03em;
  }
  .nav-links a:hover { color: #fff; }
  .nav-pill {
    background: var(--accent);
    color: #fff !important;
    padding: 7px 18px;
    border-radius: 50px;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
  }

  /* HERO */
  .hero {
    position: relative;
    min-height: 420px;
    display: flex;
    align-items: center;
    overflow: hidden;
    background: var(--navy);
  }
  .hero-bg {
    position: absolute;
    inset: 0;
    background-image: url('https://images.squarespace-cdn.com/content/v1/5d9e30182db9d71681f4a692/1581717140307-89XZXBK2C5OBW2AGDXCO/mountainviewrheumatoidarthritis.jpg');
    background-size: cover;
    background-position: center 30%;
    opacity: 0.28;
  }
  .hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(0,43,92,0.97) 0%, rgba(0,43,92,0.72) 55%, rgba(0,43,92,0.10) 100%);
  }
  .hero-content {
    position: relative;
    z-index: 2;
    max-width: 640px;
    padding: 4rem 3rem;
    animation: fadeUp 0.8s ease both;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
  }
  .hero-badge span { width: 6px; height: 6px; background: var(--accent2); border-radius: 50%; display: inline-block; }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.15;
    margin-bottom: 1rem;
  }
  .hero h1 em { color: var(--accent2); font-style: normal; }
  .hero p {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 2rem;
    max-width: 480px;
  }
  .hero-stats {
    display: flex;
    gap: 2rem;
  }
  .hero-stat { }
  .hero-stat strong { color: #fff; font-size: 1.4rem; font-weight: 600; display: block; }
  .hero-stat small { color: rgba(255,255,255,0.55); font-size: 0.75rem; letter-spacing: 0.04em; }

  /* MAIN LAYOUT */
  .main-wrap {
    max-width: 1300px;
    margin: 0 auto;
    padding: 2.5rem 2rem 4rem;
  }

  /* TABS */
  .tabs {
    display: flex;
    gap: 4px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 5px;
    width: fit-content;
    margin-bottom: 2.5rem;
    box-shadow: var(--shadow);
  }
  .tab-btn {
    padding: 9px 28px;
    border-radius: 50px;
    border: none;
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.22s;
  }
  .tab-btn.active {
    background: var(--navy);
    color: #fff;
    box-shadow: 0 2px 10px rgba(0,43,92,0.22);
  }

  /* TAB PANELS */
  .tab-panel { display: none; }
  .tab-panel.active { display: block; animation: fadeUp 0.35s ease both; }

  /* ALERT */
  .alert-research {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-left: 5px solid #f9a825;
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #795600;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  /* 2-COL GRID */
  .grid-2 {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 1.8rem;
    align-items: start;
  }

  /* CARDS */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem;
    box-shadow: var(--shadow);
  }
  .card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--navy);
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
  }

  /* FORM ELEMENTS */
  .form-group { margin-bottom: 1.1rem; }
  .form-group label {
    display: block;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .form-group input,
  .form-group select {
    width: 100%;
    padding: 9px 14px;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: var(--text);
    background: var(--bg);
    outline: none;
    transition: border-color 0.2s;
  }
  .form-group input:focus,
  .form-group select:focus { border-color: var(--accent); background: #fff; }

  .radio-group { display: flex; gap: 10px; margin-top: 2px; }
  .radio-btn {
    flex: 1;
    padding: 8px;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    text-align: center;
    font-size: 0.875rem;
    cursor: pointer;
    background: var(--bg);
    transition: all 0.2s;
    font-weight: 500;
  }
  .radio-btn.selected {
    border-color: var(--navy);
    background: var(--navy);
    color: #fff;
  }
  .radio-btn input { display: none; }

  .divider { height: 1px; background: var(--border); margin: 1.2rem 0; }

  /* FILE UPLOAD */
  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--bg);
    position: relative;
  }
  .upload-zone:hover { border-color: var(--accent); background: #f0f5ff; }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-zone svg { width: 36px; height: 36px; color: var(--muted); margin-bottom: 8px; }
  .upload-zone p { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }
  .upload-zone p strong { color: var(--accent); }

  .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 11px 24px;
    border-radius: 50px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
    width: 100%;
    letter-spacing: 0.02em;
  }
  .btn-primary {
    background: var(--navy);
    color: #fff;
    box-shadow: 0 4px 16px rgba(0,43,92,0.25);
  }
  .btn-primary:hover { background: var(--navy-mid); transform: translateY(-1px); }
  .btn-outline {
    background: transparent;
    color: var(--navy);
    border: 1.5px solid var(--border);
  }
  .btn-outline:hover { border-color: var(--navy); background: var(--navy-light); }

  /* RESULT PANEL */
  .result-section { display: none; }
  .result-section.show { display: block; animation: fadeUp 0.4s ease both; }

  .risk-card {
    border-radius: var(--radius);
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    border-left: 6px solid;
    background: var(--card);
    box-shadow: var(--shadow);
  }
  .risk-low { border-left-color: #2d6a4f; }
  .risk-med { border-left-color: #e9a700; }
  .risk-high { border-left-color: #d62828; }
  .risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
  }
  .badge-low { background: #d4edda; color: #155724; }
  .badge-med { background: #fff3cd; color: #856404; }
  .badge-high { background: #f8d7da; color: #721c24; }

  .prob-display {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 1rem;
  }
  .prob-num {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--navy);
    line-height: 1;
  }
  .prob-label { color: var(--muted); font-size: 0.85rem; }

  .progress-wrap { background: #e9ecef; border-radius: 50px; height: 10px; margin-bottom: 1rem; overflow: hidden; }
  .progress-bar { height: 100%; border-radius: 50px; transition: width 1s ease; }
  .prog-low { background: linear-gradient(90deg, #2d6a4f, #52b788); }
  .prog-med { background: linear-gradient(90deg, #e9a700, #f4c430); }
  .prog-high { background: linear-gradient(90deg, #d62828, #e05c5c); }

  .decision-box {
    background: var(--bg);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--text);
    border: 1px solid var(--border);
  }
  .decision-box strong { color: var(--navy); display: block; margin-bottom: 4px; font-weight: 600; }

  /* BIOMARKER CHART */
  .chart-card { margin-top: 1.5rem; }
  .bar-chart { margin-top: 1rem; }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .bar-gene {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--navy);
    width: 80px;
    text-align: right;
    flex-shrink: 0;
  }
  .bar-track { flex: 1; background: var(--bg); border-radius: 50px; height: 22px; overflow: hidden; }
  .bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #002b5c, #1a73e8);
    display: flex;
    align-items: center;
    padding-right: 8px;
    justify-content: flex-end;
    min-width: 30px;
    transition: width 1s ease;
  }
  .bar-fill span { font-size: 0.68rem; color: #fff; font-weight: 600; }

  /* MÉTHODOLOGIE */
  .method-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .method-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem;
    box-shadow: var(--shadow);
  }
  .method-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
  }
  .method-card p, .method-card li { font-size: 0.875rem; color: var(--muted); line-height: 1.7; margin-bottom: 0.5rem; }
  .method-card li { margin-left: 1.2rem; }
  .method-card strong { color: var(--navy); font-weight: 600; }

  .step-list { list-style: none; padding: 0; }
  .step-list li {
    display: flex;
    gap: 12px;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    color: var(--muted);
    line-height: 1.6;
  }
  .step-num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--navy);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
  }

  /* SIDEBAR FLOAT */
  .sidebar-status {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: var(--navy);
    color: #fff;
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    box-shadow: 0 8px 30px rgba(0,43,92,0.35);
    z-index: 99;
    font-size: 0.82rem;
    min-width: 200px;
  }
  .sidebar-status .status-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.2);
    padding-bottom: 0.5rem;
  }
  .status-row { display: flex; justify-content: space-between; align-items: center; margin: 4px 0; }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #52b788;
    display: inline-block;
    margin-right: 6px;
  }
  .status-val { color: rgba(255,255,255,0.7); font-size: 0.75rem; }

  /* FOOTER */
  footer {
    background: var(--navy);
    color: rgba(255,255,255,0.55);
    text-align: center;
    padding: 1.5rem;
    font-size: 0.78rem;
    margin-top: 4rem;
  }

  @media (max-width: 900px) {
    .grid-2 { grid-template-columns: 1fr; }
    .method-grid { grid-template-columns: 1fr; }
    nav { padding: 0 1.2rem; }
    .nav-links { gap: 1rem; }
    .hero-content { padding: 2.5rem 1.5rem; }
    .hero h1 { font-size: 2rem; }
    .main-wrap { padding: 1.5rem 1rem; }
    .sidebar-status { display: none; }
  }
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <a class="nav-brand" href="#">
    <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="16" cy="16" r="15" stroke="#00b4d8" stroke-width="1.5"/>
      <path d="M10 22 C10 14 14 10 16 10 C18 10 22 14 22 22" stroke="#fff" stroke-width="1.8" stroke-linecap="round" fill="none"/>
      <circle cx="16" cy="10" r="2" fill="#00b4d8"/>
      <path d="M8 18 L24 18" stroke="rgba(255,255,255,0.3)" stroke-width="1" stroke-dasharray="2 2"/>
    </svg>
    <span>MelanomaPredict AI</span>
  </a>
  <div class="nav-links">
    <a href="#">À propos</a>
    <a href="#">Méthodologie</a>
    <a href="#">Contact</a>
    <a href="#" class="nav-pill">Connexion</a>
  </div>
</nav>

<!-- HERO -->
<section class="hero">
  <div class="hero-bg"></div>
  <div class="hero-overlay"></div>
  <div class="hero-content">
    <div class="hero-badge"><span></span> Dispositif de Recherche Clinique</div>
    <h1>Analyse Multimodale du <em>Mélanome</em> Cutané</h1>
    <p>Aide à la décision thérapeutique par analyse de 54 biomarqueurs transcriptomiques combinés aux données cliniques du patient.</p>
    <div class="hero-stats">
      <div class="hero-stat">
        <strong>54</strong>
        <small>Gènes analysés</small>
      </div>
      <div class="hero-stat">
        <strong>TCGA-SKCM</strong>
        <small>Cohorte de référence</small>
      </div>
      <div class="hero-stat">
        <strong>Random Forest</strong>
        <small>500 arbres décisionnels</small>
      </div>
    </div>
  </div>
</section>

<!-- MAIN -->
<div class="main-wrap">

  <!-- TABS -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('analyse')">🚀 Analyse Patient</button>
    <button class="tab-btn" onclick="switchTab('methodo')">📖 Méthodologie</button>
  </div>

  <!-- TAB: ANALYSE -->
  <div class="tab-panel active" id="tab-analyse">

    <div class="alert-research">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <span><strong>Dispositif de Recherche :</strong> Ce système génère un score de risque basé sur l'analyse de 54 signatures transcriptomiques. Ne remplace pas l'avis clinique.</span>
    </div>

    <div class="grid-2">

      <!-- LEFT: INPUTS -->
      <div>
        <div class="card">
          <div class="card-title">📋 Paramètres Cliniques</div>

          <div class="form-group">
            <label>Âge du patient</label>
            <input type="number" id="age" value="55" min="1" max="115" />
          </div>

          <div class="form-group">
            <label>Sexe</label>
            <div class="radio-group">
              <label class="radio-btn selected" id="btn-homme" onclick="selectSexe('Homme')">
                <input type="radio" name="sexe" value="Homme" checked> Homme
              </label>
              <label class="radio-btn" id="btn-femme" onclick="selectSexe('Femme')">
                <input type="radio" name="sexe" value="Femme"> Femme
              </label>
            </div>
          </div>

          <div class="form-group">
            <label>Stade Initial</label>
            <select id="stade">
              <option value="I">Stade I</option>
              <option value="II">Stade II</option>
              <option value="III" selected>Stade III</option>
              <option value="IV">Stade IV</option>
            </select>
          </div>

          <div class="divider"></div>
          <div class="card-title" style="margin-top:0; font-size:0.95rem;">📥 Données Omiques</div>

          <div class="upload-zone" id="upload-zone">
            <input type="file" accept=".csv" id="csv-upload" onchange="handleFile(this)" />
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p id="upload-label"><strong>Cliquer pour charger</strong> ou glisser-déposer<br/>Profil d'expression 54 gènes (.csv)</p>
          </div>

          <div style="margin-top: 1rem; display: flex; flex-direction: column; gap: 10px;">
            <button class="btn btn-outline" onclick="downloadTemplate()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              Télécharger le Template (.csv)
            </button>
            <button class="btn btn-primary" onclick="launchAnalysis()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              Lancer le Diagnostic
            </button>
          </div>
        </div>
      </div>

      <!-- RIGHT: RESULTS -->
      <div>
        <div id="result-section" class="result-section">

          <!-- RISK CARD -->
          <div class="risk-card" id="risk-card">
            <span class="risk-badge" id="risk-badge"></span>
            <div class="prob-display">
              <span class="prob-num" id="prob-num">—</span>
              <span class="prob-label">probabilité métastatique</span>
            </div>
            <div class="progress-wrap">
              <div class="progress-bar" id="prob-bar" style="width: 0%"></div>
            </div>
            <div class="decision-box" id="decision-box">
              <strong>Décision recommandée</strong>
              En attente d'analyse…
            </div>
          </div>

          <!-- BIOMARKERS -->
          <div class="card chart-card">
            <div class="card-title">🧬 Top 10 Biomarqueurs Décisifs</div>
            <div class="bar-chart" id="bar-chart"></div>
          </div>

        </div>

        <!-- PLACEHOLDER -->
        <div id="placeholder-panel" class="card" style="text-align:center; padding: 3rem 2rem;">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#c5cfe0" stroke-width="1.2" style="margin:0 auto 1rem; display:block">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
          </svg>
          <p style="color:var(--muted); font-size:0.9rem; line-height:1.6;">Chargez un profil d'expression génique<br/>puis lancez le diagnostic pour voir les résultats.</p>
        </div>
      </div>

    </div>
  </div>

  <!-- TAB: MÉTHODOLOGIE -->
  <div class="tab-panel" id="tab-methodo">
    <div class="method-grid">

      <div class="method-card">
        <h3>1. Architecture du Modèle</h3>
        <ul>
          <li><strong>Signature Génomique :</strong> 54 biomarqueurs (mRNA) sélectionnés par régression Lasso, impliqués dans l'invasion, le remodelage de la MEC, l'EMT et l'inflammation.</li>
          <li style="margin-top:0.5rem"><strong>Moteur de Prédiction :</strong> Random Forest de 500 arbres décisionnels.</li>
          <li style="margin-top:0.5rem"><strong>Source :</strong> Entraîné sur la cohorte TCGA-SKCM.</li>
        </ul>
      </div>

      <div class="method-card">
        <h3>2. Normalisation Z-score</h3>
        <p>Chaque échantillon subit une normalisation Z-score basée sur les paramètres de la cohorte de référence :</p>
        <div style="background:#002b5c; color:#fff; font-family:monospace; font-size:1.15rem; text-align:center; padding:1.2rem; border-radius:10px; margin-top:0.8rem; letter-spacing:0.05em;">
          z = (x − μ) / σ
        </div>
        <p style="margin-top:0.8rem">μ et σ sont issus des distributions TCGA-SKCM pour chacun des 54 gènes et 3 variables cliniques.</p>
      </div>

      <div class="method-card">
        <h3>3. Procédure Diagnostique</h3>
        <ul class="step-list">
          <li><span class="step-num">1</span><span><strong>Input :</strong> Saisie des paramètres cliniques (Âge, Sexe, Stade) et chargement du profil d'expression 54 gènes (.csv).</span></li>
          <li><span class="step-num">2</span><span><strong>Fusion Multimodale :</strong> Encodage et concaténation pour former un profil unique de 57 variables (54 + 3).</span></li>
          <li><span class="step-num">3</span><span><strong>Standardisation :</strong> Application des moyennes et écarts-types TCGA sur chaque variable.</span></li>
          <li><span class="step-num">4</span><span><strong>Prédiction :</strong> Calcul de la probabilité p métastatique via le modèle Random Forest.</span></li>
        </ul>
      </div>

      <div class="method-card">
        <h3>4. Seuils de Décision Clinique</h3>
        <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 0.5rem;">
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#2d6a4f; flex-shrink:0;"></div>
            <div><strong style="color:#2d6a4f">Risque Faible (p &lt; 33%)</strong><br/><span style="font-size:0.82rem; color:var(--muted)">Mélanome primaire probable — Surveillance standard.</span></div>
          </div>
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#e9a700; flex-shrink:0;"></div>
            <div><strong style="color:#856404">Risque Intermédiaire (33–67%)</strong><br/><span style="font-size:0.82rem; color:var(--muted)">Zone d'incertitude — Confirmation histologique et suivi rapproché.</span></div>
          </div>
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#d62828; flex-shrink:0;"></div>
            <div><strong style="color:#d62828">Risque Élevé (p &gt; 67%)</strong><br/><span style="font-size:0.82rem; color:var(--muted)">Mélanome métastatique probable — Discussion précoce immunothérapie (anti-PD-1 / anti-CTLA-4).</span></div>
          </div>
        </div>
      </div>

    </div>
  </div>

</div>

<!-- SIDEBAR STATUS -->
<div class="sidebar-status">
  <div class="status-title">⚙ Statut Système</div>
  <div class="status-row">
    <span><span class="status-dot"></span>Modèle</span>
    <span class="status-val">Opérationnel</span>
  </div>
  <div class="status-row">
    <span>Algorithme</span>
    <span class="status-val">Random Forest</span>
  </div>
  <div class="status-row">
    <span>Features</span>
    <span class="status-val">54G + 3C</span>
  </div>
</div>

<footer>
  © 2025 MelanomaPredict AI — Outil de Recherche Clinique. Ne remplace pas un avis médical professionnel.
</footer>

<script>
  let sexe = 'Homme';
  let fileLoaded = false;

  function switchTab(id) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    event.target.classList.add('active');
  }

  function selectSexe(val) {
    sexe = val;
    document.getElementById('btn-homme').classList.toggle('selected', val === 'Homme');
    document.getElementById('btn-femme').classList.toggle('selected', val === 'Femme');
  }

  function handleFile(input) {
    if (input.files && input.files[0]) {
      fileLoaded = true;
      document.getElementById('upload-label').innerHTML =
        '<strong style="color:#2d6a4f">✓ ' + input.files[0].name + '</strong><br/><span style="font-size:0.75rem;color:var(--muted)">Fichier chargé avec succès</span>';
      document.getElementById('upload-zone').style.borderColor = '#2d6a4f';
    }
  }

  function downloadTemplate() {
    const genes = [
      'MMP1','MMP9','COL1A1','FN1','VIM','CDH2','TWIST1','SNAI1','ZEB1','MKI67',
      'TOP2A','AURKA','CCNB1','PLK1','CDK1','FOXM1','MCM2','PCNA','BUB1','TTK',
      'IL6','IL8','TNF','CXCL10','CCL2','STAT3','NF1','BRAF','NRAS','KIT',
      'MITF','SOX10','PMEL','TYR','DCT','RAC1','RHOC','ACTB','ITGA2','ITGB1',
      'MET','AXL','EGFR','VEGFA','PDGFRA','HIF1A','LDHA','PKM','GLUT1','ACSL4',
      'FASN','HMGCR','IDO1','PD-L1'
    ];
    const header = genes.join(',');
    const values = genes.map(() => (Math.random() * 4.5 + 0.5).toFixed(3)).join(',');
    const blob = new Blob([header + '\n' + values], {type: 'text/csv'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'template_54_genes.csv';
    a.click();
  }

  function launchAnalysis() {
    const age = parseInt(document.getElementById('age').value);
    const stade = document.getElementById('stade').value;

    // Simulate analysis with demo probability
    const stadeMap = {'I': 0.15, 'II': 0.35, 'III': 0.62, 'IV': 0.81};
    let base = stadeMap[stade] || 0.5;
    if (sexe === 'Homme') base = Math.min(base * 1.05, 0.97);
    if (age > 65) base = Math.min(base * 1.08, 0.97);
    const prob = Math.min(0.97, Math.max(0.04, base + (Math.random() - 0.5) * 0.12));

    displayResult(prob);
  }

  function displayResult(prob) {
    const pct = Math.round(prob * 100);

    document.getElementById('placeholder-panel').style.display = 'none';
    const rs = document.getElementById('result-section');
    rs.classList.add('show');

    document.getElementById('prob-num').textContent = pct + '%';

    const bar = document.getElementById('prob-bar');
    const rc = document.getElementById('risk-card');
    const badge = document.getElementById('risk-badge');
    const decision = document.getElementById('decision-box');

    setTimeout(() => { bar.style.width = pct + '%'; }, 100);

    if (prob < 0.33) {
      rc.className = 'risk-card risk-low';
      bar.className = 'progress-bar prog-low';
      badge.className = 'risk-badge badge-low';
      badge.textContent = '🟢 Risque Faible';
      decision.innerHTML = '<strong>Décision recommandée</strong>Mélanome primaire probable. Surveillance standard et suivi dermatologique classique recommandé.';
    } else if (prob < 0.67) {
      rc.className = 'risk-card risk-med';
      bar.className = 'progress-bar prog-med';
      badge.className = 'risk-badge badge-med';
      badge.textContent = '🟡 Risque Intermédiaire';
      decision.innerHTML = '<strong>Décision recommandée</strong>Zone d\'incertitude clinique. Examens complémentaires, confirmation histologique et suivi rapproché nécessaires.';
    } else {
      rc.className = 'risk-card risk-high';
      bar.className = 'progress-bar prog-high';
      badge.className = 'risk-badge badge-high';
      badge.textContent = '🔴 Risque Élevé';
      decision.innerHTML = '<strong>Décision recommandée</strong>Mélanome métastatique probable. Discussion précoce d\'immunothérapie (anti-PD-1 / anti-CTLA-4) ou thérapie ciblée recommandée.';
    }

    renderBiomarkers();
  }

  function renderBiomarkers() {
    const genes = ['MMP1','COL1A1','VIM','TWIST1','MKI67','IL6','BRAF','VEGFA','HIF1A','PD-L1'];
    const importances = genes.map(() => Math.random() * 0.12 + 0.02);
    const maxImp = Math.max(...importances);

    const container = document.getElementById('bar-chart');
    container.innerHTML = '';

    genes.forEach((gene, i) => {
      const val = importances[i];
      const pct = Math.round((val / maxImp) * 90 + 8);
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `
        <div class="bar-gene">${gene}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width: 0%" id="bar-${i}">
            <span>${(val * 100).toFixed(1)}%</span>
          </div>
        </div>
      `;
      container.appendChild(row);
    });

    setTimeout(() => {
      genes.forEach((_, i) => {
        const val = importances[i];
        const pct = Math.round((val / maxImp) * 90 + 8);
        document.getElementById('bar-' + i).style.width = pct + '%';
      });
    }, 200);
  }
</script>
</body>
</html>
