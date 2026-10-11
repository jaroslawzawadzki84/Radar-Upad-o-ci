import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Predykcja Bankructwa (v2.0)", layout="wide")

st.title("🏦 Aplikacja Predykcji Bankructwa Firm")
st.markdown("Wersja 2.0 (Uniwersalny model H1-H6, Kalibracja Izotoniczna)")

# 1. Wczytanie modelu i listy cech
@st.cache_resource
def load_model_and_features():
    model = joblib.load('universal_model_calibrated.pkl')
    with open('app_features.json', 'r') as f:
        features = json.load(f)
    return model, features

try:
    model, features = load_model_and_features()
except Exception as e:
    st.error(f"Nie znaleziono modelu lub pliku z cechami. Upewnij się, że uruchomiłeś skrypt trenujący. Błąd: {e}")
    st.stop()

# 2. Pasek boczny - Ustawienia Biznesowe i Horyzont
st.sidebar.header("⚙️ Ustawienia Biznesowe")

horyzont = st.sidebar.slider("Horyzont prognozy (ile lat do przodu?)", min_value=1, max_value=6, value=2)

tryb = st.sidebar.radio(
    "Dla kogo jest ta analiza? (Strategia decyzyjna)",
    options=["Bank (Ostrożnie - wysoki Recall)", "Inwestor (Ryzykownie - wysoka Precyzja)"]
)

# Przypisanie progu na podstawie trybu (przykładowe wartości, zależą od AUC modelu)
if "Bank" in tryb:
    threshold = 0.28
    st.sidebar.info(f"**Tryb Banku:** Próg alarmowy = {threshold}. Model częściej odrzuca wnioski (alarmuje), by chronić kapitał (wyłapuje 80%+ bankrutów).")
else:
    threshold = 0.70
    st.sidebar.info(f"**Tryb Inwestora:** Próg alarmowy = {threshold}. Model rzadziej podnosi alarm, szukając absolutnych pewniaków do wrogich przejęć.")

st.sidebar.markdown("---")

# 3. Formularz wprowadzania danych finansowych
st.subheader("Wprowadź dane finansowe firmy")
st.write(f"Model został zoptymalizowany, by wymagać tylko **{len(features)-1}** najważniejszych wskaźników z ponad 60 oryginalnych.")

input_data = {}

# Dynamicznie generuj pola tekstowe dla każdej cechy, poza horyzontem
cols = st.columns(3) # dzielimy formularz na 3 kolumny
i = 0
for feat in features:
    if feat == 'horizon':
        input_data[feat] = horyzont
    else:
        # Zabezpieczamy domyślną wartość na 0.0 (w prawdziwej apce można tu wstawić średnią branżową)
        with cols[i % 3]:
            input_data[feat] = st.number_input(f"Cecha: {feat}", value=0.0, step=0.1)
        i += 1

# 4. Predykcja
if st.button("🚀 Oblicz ryzyko bankructwa", type="primary"):
    # Konwersja do DataFrame o odpowiedniej kolejności kolumn
    df_input = pd.DataFrame([input_data])
    df_input = df_input[features] # upewnienie się, że kolejność jest poprawna
    
    # Przewidywanie skalibrowanym modelem
    proba = model.predict_proba(df_input)[0, 1]
    
    st.markdown("---")
    st.subheader("📊 Wynik Analizy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label=f"Prawdopodobieństwo bankructwa (w ciągu {horyzont} lat)", value=f"{proba*100:.2f}%")
        
    with col2:
        if proba >= threshold:
            st.error(f"❌ DECYZJA BIZNESOWA: ODRZUCENIE (Ryzyko {proba*100:.2f}% przekracza próg {threshold*100:.0f}%)")
        else:
            st.success(f"✅ DECYZJA BIZNESOWA: AKCEPTACJA (Ryzyko {proba*100:.2f}% poniżej progu {threshold*100:.0f}%)")
            
    # Opcjonalnie: Wyjaśnialność (SHAP)
    # Ze względu na to, że używamy CalibratedClassifierCV, wyciągnięcie SHAP z XGBoosta jest ciut trudniejsze (trzeba dobrać się do modelu bazowego w kalibratorze)
    # W uproszczonej wersji 2.0 możemy po prostu pokazać ogólną ważność cech
    
    st.info("💡 Aplikacja wykorzystuje model po kalibracji izotonicznej. Wyświetlane prawdopodobieństwo (np. 15%) oznacza, że statystycznie 15 na 100 takich firm upada (a nie tylko sztuczny score, jak w wersji 1.0).")
