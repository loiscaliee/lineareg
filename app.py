import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Forecasting BBCA", page_icon="📈")

@st.cache_resource
def load_model():
    try:
        return joblib.load('best_model_bbca.joblib')
    except:
        st.error("Model tidak ditemukan.")
        return None

model = load_model()

st.title("📈 Prediksi Saham BBCA (Simple)")
st.write("Prediksi harga penutupan saham menggunakan Linear Regression.")

with st.sidebar:
    st.header("Parameter")
    last_price = st.number_input("Harga Kemarin (Rp)", value=10200.0, step=25.0)
    days = st.slider("Jumlah Hari Prediksi", 1, 7, 3)
    predict_btn = st.button("Prediksi")

if predict_btn and model:

    future_prices = []
    current_val = last_price
    
    for _ in range(days):

        prediction = model.predict([[current_val]])[0]
        future_prices.append(prediction)
        current_val = prediction 

    df = pd.DataFrame({
        "Hari": [f"Hari ke-{i+1}" for i in range(days)],
        "Harga": future_prices
    })

    st.metric(label="Prediksi Harga Besok", value=f"Rp {future_prices[0]:,.2f}")

    chart_data = pd.DataFrame({"Harga": [last_price] + future_prices})
    st.line_chart(chart_data)

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Harga": st.column_config.NumberColumn(format="Rp %.2f")
        }
    )

    with st.expander("Rumus Model"):
        c, i = model.coef_[0], model.intercept_
        st.latex(f"Harga_t = (Harga_{{t-1}} \\times {c:.4f}) + {i:.4f}")

elif not predict_btn:
    st.info("👈 Masukkan data di sidebar dan klik 'Prediksi'.")
