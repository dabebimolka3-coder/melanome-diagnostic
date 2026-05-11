<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NavBar Professionnelle | MedFlow Style</title>
    <style>
        /* Reset & Base */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(145deg, #f6f9fc 0%, #edf2f7 100%);
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, sans-serif;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }

        /* Carte principale style dashboard */
        .dashboard-card {
            max-width: 1200px;
            width: 100%;
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(2px);
            border-radius: 28px;
            box-shadow: 0 20px 35px -12px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(0, 0, 0, 0.02);
            overflow: hidden;
            transition: all 0.3s ease;
        }

        /* ========== NAVIGATION PRINCIPALE – STYLE MÉDICAL HAUT DE GAMME ========== */
        .nav-container {
            background: #ffffff;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            padding: 0 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }

        /* Logo / Marque */
        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.8rem 0;
        }
        .brand-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #1E6DFF, #00B8A9);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            box-shadow: 0 6px 14px rgba(30, 109, 255, 0.2);
        }
        .brand-name {
            font-weight: 600;
            font-size: 1.25rem;
            letter-spacing: -0.3px;
            background: linear-gradient(135deg, #111B2C, #1E3A5F);
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
        }
        .brand-badge {
            font-size: 0.7rem;
            background: #EEF2FF;
            color: #1E6DFF;
            padding: 4px 8px;
            border-radius: 40px;
            font-weight: 500;
            margin-left: 8px;
        }

        /* Liste de navigation stylée */
        .nav-links {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            list-style: none;
        }

        /* Élément de navigation individuel – version ultra professionnelle */
        .nav-item {
            position: relative;
        }
        .nav-link {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0.6rem 1.2rem;
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.02em;
            color: #4A5B6E;
            text-decoration: none;
            border-radius: 40px;
            transition: all 0.25s ease;
            background: transparent;
            cursor: pointer;
            border: none;
            font-family: inherit;
        }

        /* Icônes incluses (FontAwesome via CDN) */
        .nav-link i {
            font-size: 1rem;
            opacity: 0.7;
            transition: opacity 0.2s;
        }

        /* Effet hover professionnel */
        .nav-link:hover {
            background: #F0F4F9;
            color: #0F2B3D;
        }
        .nav-link:hover i {
            opacity: 1;
            color: #1E6DFF;
        }

        /* Style actif (page courante) – très chic */
        .nav-link.active {
            background: #EFF4FF;
            color: #1A56DB;
            box-shadow: inset 0 0 0 1px rgba(30, 109, 255, 0.2), 0 1px 2px rgba(0,0,0,0.02);
        }
        .nav-link.active i {
            opacity: 1;
            color: #1A56DB;
        }

        /* Indicateur de statut "système actif" */
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #F8FAFE;
            padding: 6px 14px;
            border-radius: 40px;
            font-size: 0.7rem;
            font-weight: 500;
            color: #2C7A6E;
            border: 1px solid rgba(0,184,169,0.2);
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background: #00B8A9;
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(0,184,169,0.5);
            animation: pulse-medical 1.8s infinite;
        }
        @keyframes pulse-medical {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,184,169,0.6);}
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0,184,169,0);}
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0,184,169,0);}
        }

        /* Contenu d'exemple */
        .content-demo {
            padding: 3rem 2rem;
            background: #FFFFFF;
        }
        .hero-mini {
            background: linear-gradient(115deg, #F8FBFE 0%, #FFFFFF 100%);
            border-radius: 24px;
            padding: 2rem;
            border: 1px solid rgba(0,0,0,0.04);
            box-shadow: 0 5px 18px rgba(0,0,0,0.02);
        }
        h2 {
            font-size: 1.7rem;
            font-weight: 600;
            background: linear-gradient(135deg, #152C42, #1E5A7A);
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: 0.5rem;
        }
        .badge-demo {
            display: inline-block;
            background: #EFF6FF;
            border-radius: 40px;
            padding: 0.2rem 0.8rem;
            font-size: 0.7rem;
            font-weight: 600;
            color: #1E6DFF;
            margin-top: 0.5rem;
        }
        hr {
            margin: 1rem 0;
            border: none;
            border-top: 1px solid #E9EDF2;
        }

        /* Responsive */
        @media (max-width: 750px) {
            .nav-container {
                flex-direction: column;
                align-items: stretch;
                padding: 1rem;
            }
            .nav-links {
                justify-content: space-between;
                flex-wrap: wrap;
            }
            .nav-link {
                padding: 0.5rem 0.9rem;
                font-size: 0.75rem;
            }
            .status-indicator {
                align-self: flex-start;
            }
        }
    </style>
    <!-- Font Awesome 6 (gratuit, icônes pro) -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
</head>
<body>
<div class="dashboard-card">
    <!-- Barre de navigation professionnelle MEDFLOW / MELANOMA AI -->
    <div class="nav-container">
        <div class="brand">
            <div class="brand-icon">
                <i class="fas fa-dna" style="color: white; font-size: 18px;"></i>
            </div>
            <div>
                <span class="brand-name">MelanomaPredict AI</span>
                <span class="brand-badge">Research Grade</span>
            </div>
        </div>

        <ul class="nav-links">
            <!-- MÉTHODOLOGIE : élégant, avec icône de recherche/flux -->
            <li class="nav-item">
                <a href="#" class="nav-link" data-page="methodologie">
                    <i class="fas fa-microscope"></i>
                    <span>MÉTHODOLOGIE</span>
                </a>
            </li>
            <!-- DOCUMENTATION : style scientifique -->
            <li class="nav-item">
                <a href="#" class="nav-link" data-page="documentation">
                    <i class="fas fa-book-open"></i>
                    <span>DOCUMENTATION</span>
                </a>
            </li>
            <!-- CONTACT : médical / support premium -->
            <li class="nav-item">
                <a href="#" class="nav-link" data-page="contact">
                    <i class="fas fa-headset"></i>
                    <span>CONTACT</span>
                </a>
            </li>
        </ul>

        <div class="status-indicator">
            <div class="pulse-dot"></div>
            <span>Système opérationnel · API v2.3</span>
            <i class="fas fa-shield-alt" style="font-size: 0.7rem; opacity: 0.6;"></i>
        </div>
    </div>

    <!-- Zone de contenu dynamique (simulation) -->
    <div class="content-demo" id="mainContent">
        <div class="hero-mini">
            <i class="fas fa-chalkboard-user" style="font-size: 2rem; color: #1E6DFF; margin-bottom: 0.5rem; display: inline-block;"></i>
            <h2>Méthodologie clinique & IA</h2>
            <p>Modèle multimodal Random Forest (500 arbres) · 54 biomarqueurs transcriptomiques + staging TNM.<br>
            Validation croisée stratifiée, AUC ROC 0.955. Cohorte TCGA-SKCM de référence.</p>
            <div class="badge-demo"><i class="fas fa-chart-line"></i> Fiabilité diagnostique 94%</div>
        </div>
    </div>
</div>

<script>
    // Simulation navigation haut de gamme – switching de contenu avec animations fluides
    // Mapping des contenus pour les onglets MÉTHODOLOGIE, DOCUMENTATION, CONTACT
    const contentMap = {
        methodologie: `
            <div class="hero-mini">
                <i class="fas fa-flask" style="font-size: 2rem; color: #1E6DFF; margin-bottom: 0.5rem;"></i>
                <h2>Architecture du modèle & validation</h2>
                <p><strong>Random Forest</strong> · 57 features (54 gènes + 3 cliniques). Normalisation Z-score sur cohorte TCGA.<br>
                Seuils cliniques : Risque faible (<33%), intermédiaire (33-67%), élevé (>67%).<br>
                Interprétabilité via importance SHAP intégrée.</p>
                <div class="badge-demo"><i class="fas fa-chart-simple"></i> 10-fold CV · Sensibilité 85%</div>
                <hr>
                <p><i class="fas fa-database" style="color:#1E6DFF;"></i> <strong>Processus :</strong> Fusion multimodal, standardisation, prédiction proba métastatique.</p>
            </div>
        `,
        documentation: `
            <div class="hero-mini">
                <i class="fas fa-file-alt" style="font-size: 2rem; color: #1E6DFF; margin-bottom: 0.5rem;"></i>
                <h2>Documentation scientifique</h2>
                <p>Référentiel complet du dispositif de recherche :<br>
                • TCGA-SKCM (n=473)<br>
                • Sélection des gènes par ElasticNet & stabilité clinique<br>
                • Métriques détaillées : précision 94.44%, F1-score 89.5%<br>
                • Notebooks d'analyse et API REST (OpenAPI)</p>
                <div class="badge-demo"><i class="fas fa-graduation-cap"></i> DOI: 10.xxxx/melanomapredict.2026</div>
            </div>
        `,
        contact: `
            <div class="hero-mini">
                <i class="fas fa-envelope" style="font-size: 2rem; color: #1E6DFF; margin-bottom: 0.5rem;"></i>
                <h2>Contact & support recherche</h2>
                <p>Équipe d'oncologie numérique — CHU & laboratoire de bioinformatique translationnelle.<br>
                <i class="fas fa-inbox"></i> <strong>contact@melanomapredict.ai</strong><br>
                <i class="fas fa-phone-alt"></i> +33 (0)1 70 98 55 00 (standard recherche)<br>
                Formulaire de collaboration clinique disponible sur demande.</p>
                <div class="badge-demo"><i class="fas fa-lock"></i> Conformité RGPD / hébergement HDS</div>
            </div>
        `
    };

    // Récupération des éléments
    const navLinks = document.querySelectorAll('.nav-link');
    const contentDiv = document.getElementById('mainContent');

    // Fonction pour mettre à jour l'onglet actif et le contenu
    function setActivePage(pageId) {
        // Mise à jour des classes active sur les liens
        navLinks.forEach(link => {
            const page = link.getAttribute('data-page');
            if (page === pageId) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
        // Mise à jour du contenu avec transition douce
        if (contentMap[pageId]) {
            contentDiv.style.opacity = '0';
            setTimeout(() => {
                contentDiv.innerHTML = contentMap[pageId];
                contentDiv.style.transition = 'opacity 0.2s ease';
                contentDiv.style.opacity = '1';
            }, 120);
        } else {
            // fallback
            contentDiv.innerHTML = `<div class="hero-mini"><p>Contenu non disponible</p></div>`;
        }
    }

    // Ajout des événements de clic pour chaque onglet
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.getAttribute('data-page');
            if (page) {
                setActivePage(page);
                // Optionnel : mise à jour de l'URL sans rechargement (pushState)
                history.pushState({ page: page }, '', `?tab=${page}`);
            }
        });
    });

    // Vérification des paramètres d'URL pour afficher l'onglet correct au chargement
    const urlParams = new URLSearchParams(window.location.search);
    const tabParam = urlParams.get('tab');
    if (tabParam && contentMap[tabParam]) {
        setActivePage(tabParam);
    } else {
        // Par défaut, activer 'methodologie' (en adéquation avec l'aperçu)
        setActivePage('methodologie');
    }

    // Gestion du bouton retour / avant du navigateur
    window.addEventListener('popstate', (event) => {
        const state = event.state;
        if (state && state.page && contentMap[state.page]) {
            setActivePage(state.page);
        } else {
            const params = new URLSearchParams(window.location.search);
            const pageFallback = params.get('tab');
            if (pageFallback && contentMap[pageFallback]) setActivePage(pageFallback);
            else setActivePage('methodologie');
        }
    });
</script>
</body>
</html>
