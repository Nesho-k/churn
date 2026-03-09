# -*- coding: utf-8 -*-
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from src.serving.inference import predict

st.set_page_config(
    page_title="Prédiction de Résiliation Client",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body { font-family: 'Inter', sans-serif; }
    .block-container { padding: 0 3rem 3rem 3rem; }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8C00 100%);
        padding: 2.5rem 3rem 0 3rem;
        margin: -3rem -3rem 2.5rem -3rem;
        color: white;
    }
    .header-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 2rem;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .header-sub {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
    }
    .header-stats {
        display: flex;
        gap: 1px;
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        overflow: hidden;
        align-self: center;
    }
    .stat-card {
        background: rgba(255,255,255,0.12);
        padding: 0.9rem 1.6rem;
        text-align: center;
        min-width: 110px;
    }
    .stat-card-value {
        font-size: 1.3rem;
        font-weight: 800;
        display: block;
    }
    .stat-card-label {
        font-size: 0.7rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .header-line {
        height: 3px;
        background: rgba(255,255,255,0.25);
        margin: 0 -3rem;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #FF6B00;
        border-left: 3px solid #FF6B00;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* ── Results ── */
    .result-churn {
        background: linear-gradient(135deg, #C0392B, #E74C3C);
        color: white;
        padding: 2rem;
        border-radius: 14px;
        text-align: center;
    }
    .result-churn .result-title { font-size: 1.6rem; font-weight: 700; }
    .result-churn .result-sub { font-size: 0.95rem; opacity: 0.88; margin-top: 0.5rem; }

    .result-safe {
        background: linear-gradient(135deg, #1E8449, #27AE60);
        color: white;
        padding: 2rem;
        border-radius: 14px;
        text-align: center;
    }
    .result-safe .result-title { font-size: 1.6rem; font-weight: 700; }
    .result-safe .result-sub { font-size: 0.95rem; opacity: 0.88; margin-top: 0.5rem; }

    /* ── Button ── */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #FF6B00, #FF8C00) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.75rem !important;
        letter-spacing: 0.3px;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        opacity: 0.92 !important;
        box-shadow: 0 4px 14px rgba(255, 107, 0, 0.4) !important;
    }

    /* ── Progress bar color ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B00, #FF8C00);
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <div class="header-top">
        <div>
            <h1 class="header-title">Prédiction de Résiliation Client</h1>
            <p class="header-sub">Analysez le risque de résiliation grâce à un modèle XGBoost entraîné sur des données telecom réelles.</p>
        </div>
        <div class="header-stats">
            <div class="stat-card">
                <span class="stat-card-value">XGBoost</span>
                <span class="stat-card-label">Modèle</span>
            </div>
            <div class="stat-card">
                <span class="stat-card-value">19</span>
                <span class="stat-card-label">Variables</span>
            </div>
            <div class="stat-card">
                <span class="stat-card-value">7 043</span>
                <span class="stat-card-label">Clients</span>
            </div>
        </div>
    </div>
    <div class="header-line"></div>
</div>
""", unsafe_allow_html=True)

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-label">Profil client</div>', unsafe_allow_html=True)
        gender = st.selectbox("Genre", ["Male", "Female"],
            format_func=lambda x: "Homme" if x == "Male" else "Femme")
        senior = st.selectbox("Senior (65 ans et plus)", [0, 1],
            format_func=lambda x: "Non" if x == 0 else "Oui")
        partner = st.selectbox("En couple", ["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non")
        dependents = st.selectbox("Personnes à charge", ["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non")

        st.markdown('<div class="section-label">Facturation</div>', unsafe_allow_html=True)
        tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=12)
        monthly_charges = st.number_input("Charges mensuelles (€)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total_charges = st.number_input("Charges totales (€)", min_value=0.0, max_value=10000.0, value=780.0, step=1.0)

    with col2:
        st.markdown('<div class="section-label">Services téléphoniques</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Service téléphonique", ["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non")
        multiple_lines = st.selectbox("Lignes multiples", ["Yes", "No", "No phone service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No phone service": "Sans service"}[x])

        st.markdown('<div class="section-label">Services internet</div>', unsafe_allow_html=True)
        internet_service = st.selectbox("Type de connexion", ["DSL", "Fiber optic", "No"],
            format_func=lambda x: {"DSL": "DSL", "Fiber optic": "Fibre optique", "No": "Aucun"}[x])
        online_security = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No internet service": "Sans internet"}[x])
        online_backup = st.selectbox("Sauvegarde en ligne", ["Yes", "No", "No internet service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No internet service": "Sans internet"}[x])
        device_protection = st.selectbox("Protection des appareils", ["Yes", "No", "No internet service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No internet service": "Sans internet"}[x])
        tech_support = st.selectbox("Support technique", ["Yes", "No", "No internet service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No internet service": "Sans internet"}[x])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No internet service": "Sans internet"}[x])
        streaming_movies = st.selectbox("Streaming Films", ["Yes", "No", "No internet service"],
            format_func=lambda x: {"Yes": "Oui", "No": "Non", "No internet service": "Sans internet"}[x])

    with col3:
        st.markdown('<div class="section-label">Contrat</div>', unsafe_allow_html=True)
        contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"],
            format_func=lambda x: {"Month-to-month": "Mensuel", "One year": "Un an", "Two year": "Deux ans"}[x])
        paperless_billing = st.selectbox("Facturation électronique", ["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non")
        payment_method = st.selectbox("Mode de paiement",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            format_func=lambda x: {
                "Electronic check": "Chèque électronique",
                "Mailed check": "Chèque postal",
                "Bank transfer (automatic)": "Virement automatique",
                "Credit card (automatic)": "Carte bancaire automatique"
            }[x])

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        submitted = st.form_submit_button("Analyser", use_container_width=True)

if submitted:
    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    _, col_result, _ = st.columns([0.5, 2, 0.5])
    with col_result:
        bar = st.progress(0, text="Préparation des données...")
        time.sleep(0.3)
        bar.progress(40, text="Application du modèle...")
        result = predict(data)
        time.sleep(0.3)
        bar.progress(100, text="Analyse terminée")
        time.sleep(0.3)
        bar.empty()

        if result == "Likely to churn":
            st.markdown("""
            <div class="result-churn">
                <div class="result-title">Risque élevé de résiliation</div>
                <div class="result-sub">Ce client est susceptible de résilier son abonnement. Une action commerciale est recommandée.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <div class="result-title">Faible risque de résiliation</div>
                <div class="result-sub">Ce client est susceptible de rester fidèle. Aucune action urgente requise.</div>
            </div>
            """, unsafe_allow_html=True)
