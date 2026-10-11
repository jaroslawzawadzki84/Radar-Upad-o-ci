import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="AI Bankruptcy Predictor", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (Styl a'la niemiecki dashboard ze screena) ---
st.markdown("""
<style>
    /* Główne tło */
    .stApp {
        background-color: #0b0f19;
        color: white;
    }
    
    /* Ukrycie standardowych elementów Streamlita */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Wielki, fioletowo-niebieski baner u góry */
    .gradient-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    .gradient-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .gradient-header p {
        color: #d1d5db;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Karty (szklany efekt - glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Przyciski */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    
    /* Poprawa czytelności inputów tekstowych na ciemnym tle */
    .stNumberInput label {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ŁADOWANIE NASZEGO UNIWERSALNEGO MODELU ---
@st.cache_resource
def load_model():
    model = joblib.load('universal_model_calibrated.pkl')
    with open('app_features.json', 'r') as f:
        features = json.load(f)
    return model, features

try:
    model, all_features = load_model()
except Exception as e:
    st.error("Brak pliku universal_model_calibrated.pkl! Uruchom najpierw build_universal_model.py")
    st.stop()

# --- HEADER ---
st.markdown("""
<div class="gradient-header">
    <h1>System Predykcji Bankructwa (AI)</h1>
    <p>Oparty na zoptymalizowanym, hybrydowym modelu XGBoost z kalibracją izotoniczną. 
    Wprowadź dane firmy, aby natychmiastowo ocenić ryzyko upadku.</p>
</div>
""", unsafe_allow_html=True)


# --- DANE MOCKOWE DO DEMO ---
# Symulowane dane dla firm, żeby nie musieć za każdym razem generować ich ze starych, nieistniejących plików
demo_healthy = {f: np.random.uniform(0.5, 1.5) for f in all_features}
demo_healthy['horizon'] = 1
demo_healthy[all_features[0]] = 2.5 # dobre aktywa
demo_healthy[all_features[1]] = 1.2 # dobre zyski

demo_bankrupt = {f: np.random.uniform(-1.0, 0.5) for f in all_features}
demo_bankrupt['horizon'] = 1
demo_bankrupt[all_features[0]] = -0.5 # brak płynności
demo_bankrupt[all_features[1]] = -2.1 # długi


st.markdown("<div class='glass-card'><h3>🎭 Szybkie Demo (Zaladuj gotowy profil)</h3></div>", unsafe_allow_html=True)
col_d1, col_d2 = st.columns(2)
with col_d1:
    if st.button("🟢 Wczytaj Profil: Apple / Microsoft (Zdrowy gigant)", use_container_width=True):
        for f in all_features:
            st.session_state[f"in_{f}"] = demo_healthy[f]
        st.rerun()
with col_d2:
    if st.button("🔴 Wczytaj Profil: Firma XYZ (Przed bankructwem)", use_container_width=True):
        for f in all_features:
            st.session_state[f"in_{f}"] = demo_bankrupt[f]
        st.rerun()


# --- FORMULARZ (Tylko 10 cech) ---
st.markdown("<div class='glass-card'><h3>📝 Główne Wskaźniki Finansowe</h3>", unsafe_allow_html=True)

# Wybieramy tylko pierwsze 10 cech do wyświetlenia, żeby nie straszyć klienta
visible_features = [f for f in all_features if f != 'horizon'][:10]

input_vals = {}
cols = st.columns(2)

for i, f in enumerate(visible_features):
    val = st.session_state.get(f"in_{f}", 0.0)
    with cols[i % 2]:
        input_vals[f] = st.number_input(f"Wskaźnik: {f[:30]}...", value=float(val), format="%.4f", key=f"in_ui_{f}")

st.markdown("</div>", unsafe_allow_html=True)

# Zbierz wszystkie 15 cech do predykcji (te 5 ukrytych ładujemy z sesji lub jako 0)
final_input = {}
for f in all_features:
    if f == 'horizon':
        final_input[f] = 1 # Domyślnie badamy na 1 rok do przodu
    elif f in visible_features:
        final_input[f] = input_vals[f]
    else:
        final_input[f] = st.session_state.get(f"in_{f}", 0.0)


# --- PREDYKCJA I WYKRESY ---
st.markdown("<div class='glass-card'><h3>📈 Wynik Analizy AI</h3>", unsafe_allow_html=True)

if st.button("🚀 Uruchom Silnik AI", type="primary", use_container_width=True):
    df_pred = pd.DataFrame([final_input])[all_features]
    proba = model.predict_proba(df_pred)[0, 1]
    
    st.markdown("---")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if proba > 0.5:
            st.error("🚨 WYSOKIE RYZYKO BANKRUCTWA")
        else:
            st.success("✅ FIRMA STABILNA")
            
        st.metric("Prawdopodobieństwo upadku", f"{proba*100:.1f}%")
        
    with col_res2:
        # Rysujemy piękny wykres wskaźnikowy (Gauge) za pomocą Matplotlib
        fig, ax = plt.subplots(figsize=(6, 2), facecolor='#1e293b')
        ax.axis('off')
        
        # Pasek tła
        ax.barh(0, 100, height=0.5, color='#334155', align='center')
        # Pasek wyniku (Zmienia kolor od zielonego do czerwonego)
        color = 'red' if proba > 0.5 else 'green'
        ax.barh(0, proba*100, height=0.5, color=color, align='center')
        
        ax.text(50, 0.4, "Skala Ryzyka (0 - 100%)", color='white', ha='center', fontsize=12)
        ax.text(proba*100, 0, f" {proba*100:.1f}%", color=color, va='center', fontweight='bold', fontsize=14)
        
        st.pyplot(fig)
        
st.markdown("</div>", unsafe_allow_html=True)
