import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Forecasting Saham BBCA - Group 6",
    page_icon="📈",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
  
        model = joblib.load('best_model_bbca.joblib')
        return model
    except FileNotFoundError:
        st.error("File 'best_model_bbca.joblib' tidak ditemukan. Pastikan file model ada di folder yang sama.")
        return None

model = load_model()

st.title("📈 Prediksi Harga Saham BBCA")
st.markdown("""
Aplikasi ini menggunakan Machine Learning (**Linear Regression**) untuk memprediksi harga penutupan saham BBCA 
berdasarkan harga penutupan hari sebelumnya.

**Dibuat oleh Group 6 - IS388**
""")
st.divider()

st.sidebar.header("Parameter Input")

last_price = st.sidebar.number_input(
    "Harga Penutupan Kemarin (Rp)", 
    min_value=0.0, 
    value=10200.0, 
    step=25.0,
    format="%.2f"
)

days_to_predict = st.sidebar.slider("Jumlah Hari Prediksi", 1, 7, 1)

predict_btn = st.sidebar.button("Prediksi Sekarang")

if predict_btn and model is not None:
    st.subheader(f"Hasil Prediksi untuk {days_to_predict} Hari Ke Depan")
    
    future_predictions = []
    current_input = np.array([[last_price]]) # Format array 2D untuk Scikit-Learn
    
    for i in range(days_to_predict):
        prediction = model.predict(current_input)[0]
        future_predictions.append(prediction)
        
        # Update input untuk iterasi berikutnya (rekursif)
        current_input = np.array([[prediction]])

    # Tampilkan Hasil dalam Tabel
    forecast_dates = [f"Hari ke-{i+1}" for i in range(days_to_predict)]
    df_result = pd.DataFrame({
        "Hari": forecast_dates,
        "Prediksi Harga (Rp)": future_predictions
    })
    
    # Formatting angka agar rapi
    df_result["Prediksi Harga (Rp)"] = df_result["Prediksi Harga (Rp)"].apply(lambda x: f"Rp {x:,.2f}")

    # Layout Kolom untuk hasil
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Tabel Hasil:")
        st.dataframe(df_result, hide_index=True)
        
        # Highlight prediksi besok
        st.success(f"Harga Besok: **{df_result.iloc[0]['Prediksi Harga (Rp)']}**")

    with col2:
        # Visualisasi Grafik
        fig = go.Figure()
        
        # Titik awal (Hari 0 / Kemarin)
        fig.add_trace(go.Scatter(
            x=[0], 
            y=[last_price],
            mode='markers+text',
            name='Harga Awal',
            text=[f"{last_price:,.0f}"],
            textposition="bottom center",
            marker=dict(color='gray', size=10)
        ))

        # Garis Prediksi
        fig.add_trace(go.Scatter(
            x=list(range(1, days_to_predict + 1)), 
            y=future_predictions,
            mode='lines+markers+text',
            name='Prediksi',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            text=[f"{p:,.0f}" for p in future_predictions],
            textposition="top center"
        ))

        fig.update_layout(
            title="Grafik Pergerakan Harga Prediksi",
            xaxis_title="Hari ke-",
            yaxis_title="Harga (Rp)",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Info Model
    with st.expander("Lihat Detail Model"):
        st.write(f"**Model Type:** Linear Regression")
        st.write(f"**Coefficient (Kemiringan):** {model.coef_[0]:.4f}")
        st.write(f"**Intercept (Konstanta):** {model.intercept_:.4f}")
        st.latex(r"Harga_{t} = (Harga_{t-1} \times " + f"{model.coef_[0]:.4f}) + {model.intercept_:.4f}")

else:
    if model is not None:
        st.info("👈 Masukkan harga penutupan terakhir di menu sebelah kiri dan klik 'Prediksi Sekarang'.")
