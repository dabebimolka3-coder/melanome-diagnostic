# ─────────────────────────────────────────────────────────────
# TOPBAR FONCTIONNELLE STREAMLIT
# ─────────────────────────────────────────────────────────────

# CSS À AJOUTER DANS TON <style>
st.markdown("""
<style>

.topbar-wrapper {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(0,0,0,0.08);
    padding: 0.7rem 0.5rem;
    margin: 0 -2.5rem 1.5rem;
}

.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.topbar-logo {
    width: 34px;
    height: 34px;
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

.topbar-status {
    display: flex;
    align-items: center;
    justify-content: flex-end;
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

/* STYLE DES BOUTONS NAVIGATION */

.nav-btn div button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: rgba(0,0,0,0.45) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    padding-top: 0.2rem !important;
}

.nav-btn div button:hover {
    color: #1a6fff !important;
    transform: none !important;
}

/* ACTIVE PAGE */

.active-nav div button {
    color: #1a6fff !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="topbar-wrapper">', unsafe_allow_html=True)

col_logo, col_nav, col_status = st.columns([2,6,2])

# ── LOGO ──────────────────────────────────────
with col_logo:
    st.markdown("""
    <div class="topbar-brand">
        <div class="topbar-logo">🧬</div>
        <span class="topbar-name">MelanomaPredict AI</span>
    </div>
    """, unsafe_allow_html=True)

# ── NAVIGATION ────────────────────────────────
with col_nav:

    nav1, sep1, nav2, sep2, nav3, sep3, nav4 = st.columns(
        [2,0.3,2,0.3,2,0.3,2]
    )

    # ANALYSE
    with nav1:
        cls = "active-nav" if st.session_state['current_page'] == "analyse" else "nav-btn"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

        if st.button("ANALYSE", key="top_analyse", use_container_width=True):
            st.session_state['current_page'] = 'analyse'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with sep1:
        st.markdown(
            "<div style='text-align:center;color:rgba(0,0,0,0.2);padding-top:0.7rem;'>|</div>",
            unsafe_allow_html=True
        )

    # MÉTHODOLOGIE
    with nav2:
        cls = "active-nav" if st.session_state['current_page'] == "methodologie" else "nav-btn"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

        if st.button("MÉTHODOLOGIE", key="top_methodo", use_container_width=True):
            st.session_state['current_page'] = 'methodologie'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with sep2:
        st.markdown(
            "<div style='text-align:center;color:rgba(0,0,0,0.2);padding-top:0.7rem;'>|</div>",
            unsafe_allow_html=True
        )

    # DOCUMENTATION
    with nav3:
        cls = "active-nav" if st.session_state['current_page'] == "documentation" else "nav-btn"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

        if st.button("DOCUMENTATION", key="top_doc", use_container_width=True):
            st.session_state['current_page'] = 'documentation'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with sep3:
        st.markdown(
            "<div style='text-align:center;color:rgba(0,0,0,0.2);padding-top:0.7rem;'>|</div>",
            unsafe_allow_html=True
        )

    # CONTACT
    with nav4:
        cls = "active-nav" if st.session_state['current_page'] == "contact" else "nav-btn"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

        if st.button("CONTACT", key="top_contact", use_container_width=True):
            st.session_state['current_page'] = 'contact'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ── STATUS ────────────────────────────────────
with col_status:
    st.markdown("""
    <div class="topbar-status">
        <span class="pulse"></span>
        Système opérationnel
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
