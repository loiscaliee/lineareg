import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Forecasting BBCA", page_icon="📈")

@st.cache_resource
def load_model():
    try:
        return joblib.load('best_model_bbca.joblib')
    except:
        st.error("File model 'best_model_bbca.joblib' tidak ditemukan.")
        return None

model = load_model()

st.title("📈 Prediksi Harga Saham BBCA")
st.write("Aplikasi ini memprediksi harga saham besok berdasarkan harga hari ini.")

st.divider()

current_price = st.number_input(
    "Harga Penutupan Hari Ini (Rp):", 
    min_value=0.0, 
    value=10200.0, 
    step=25.0
)

if st.button("Prediksi Sekarang") and model:

    prediction = model.predict([[current_price]])[0]

    selisih = prediction - current_price
    persentase = (selisih / current_price) * 100

    st.subheader("Hasil Prediksi")
    
    st.metric(
        label="Prediksi Harga Besok",
        value=f"Rp {prediction:,.2f}",
        delta=f"{selisih:,.2f} ({persentase:.2f}%)"
    )

    st.subheader("Grafik Perbandingan")
    
    chart_data = pd.DataFrame({
        "Harga": [current_price, prediction]
    }, index=["Hari Ini", "Besok (Prediksi)"])
    
    st.bar_chart(chart_data)

elif not model:
    st.info("Model belum siap digunakan.")
