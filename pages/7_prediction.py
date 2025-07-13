# pages/14_Auto_Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import utils_for_made_model
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Otomatis", page_icon="🤖", layout="wide")
st.title("🤖 Prediksi Otomatis Berdasarkan Data Terakhir")

st.info("""
Halaman ini membuat peramalan masa depan dengan asumsi bahwa kondisi akan sama seperti beberapa jam terakhir yang tercatat dalam dataset. 
Pilih periode waktu yang Anda inginkan untuk melihat prediksinya.
""", icon="💡")

# --- PATH & PEMUATAN MODEL ---
MODEL_DIR = "models"
FULL_MODEL_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "full_prediction_artifacts.joblib")

artifacts = utils_for_made_model.load_prediction_artifacts(FULL_MODEL_ARTIFACTS_PATH)

if not artifacts:
    st.error("Model final (data penuh) belum dilatih. Harap latih model terlebih dahulu di halaman 'Prediksi (Data Penuh)'.")
    st.page_link("pages/13_prediction_full.py", label="Buka Halaman Pelatihan Model", icon="🔮")
    st.stop()

st.success(f"Model final (data penuh) dengan order **{artifacts['final_order']}** berhasil dimuat.", icon="✅")
final_model = artifacts['model']
scaler = artifacts['scaler']
final_features = artifacts['final_features']

if 'main_dataframe_full' not in st.session_state:
    st.session_state.main_dataframe_full = utils_for_made_model.load_station_data('Aotizhongxin')

# --- ANTARMUKA PENGGUNA (DIPERBARUI) ---
st.header("Form Prediksi", divider="blue")

# Definisikan opsi periode
period_options = {
    "1 Hari (24 Jam)": 24,
    "3 Hari (72 Jam)": 72,
    "1 Minggu (168 Jam)": 168,
    "2 Minggu (336 Jam)": 336,
    "1 Bulan (720 Jam)": 720,
    "3 Bulan (2160 Jam)": 2160,
    "6 Bulan (4320 Jam)": 4320,
    "9 Bulan (6480 Jam)": 6480,
    "1 Tahun (8640 Jam)": 8640,
    "1,5 Tahun (12960 Jam)": 12960,
    "2 Tahun (17280 Jam)": 17280,
    "3 Tahun (21600 Jam)": 21600,
    "Prediksi penuh": len(st.session_state.main_dataframe_full)
}

selected_period = st.selectbox(
    "Pilih Periode Prediksi Masa Depan:",
    options=list(period_options.keys()),
    help="Jumlah jam ini juga akan menentukan data historis mana yang digunakan sebagai asumsi nilai eksogen."
)

if st.button("Buat Prediksi Otomatis", type="primary", use_container_width=True):
    n_periods = period_options[selected_period]
    
    with st.spinner(f"Membuat peramalan untuk {selected_period}..."):
        exog_input_df = st.session_state.main_dataframe_full[final_features].tail(n_periods).copy()
        exog_input_df.reset_index(drop=True, inplace=True)

        with st.expander("Lihat Asumsi Nilai Eksogen yang Digunakan"):
            st.dataframe(exog_input_df)

        original_predictions, scaled_predictions = utils_for_made_model.predict_future_values(
            final_model=final_model,
            scaler=scaler,
            exog_input_df=exog_input_df,
            final_features=final_features
        )
        
        last_timestamp = st.session_state.main_dataframe_full.index[-1]
        future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')
        
        result_df = pd.DataFrame({"Waktu Prediksi": future_index, "Prediksi PM2.5 (µg/m³)": original_predictions})
        st.subheader("Hasil Peramalan")
        st.dataframe(result_df.style.format({"Waktu Prediksi": "{:%Y-%m-%d %H:%M}", "Prediksi PM2.5 (µg/m³)": "{:.2f}"}), use_container_width=True)

        st.subheader("Grafik Prediksi Kontekstual")
        historical_data_unscaled = st.session_state.main_dataframe_full['PM2.5'].tail(n_periods * 2) # Tampilkan histori 2x lipat dari prediksi
        fig = utils_for_made_model.go.Figure()
        fig.add_trace(utils_for_made_model.go.Scatter(x=historical_data_unscaled.index, y=historical_data_unscaled, mode='lines', name='Histori PM2.5 (Skala Asli)'))
        fig.add_trace(utils_for_made_model.go.Scatter(x=future_index, y=original_predictions, mode='lines+markers', name='Prediksi PM2.5 (Skala Asli)', line=dict(color='red')))
        fig.update_layout(title="Visualisasi Prediksi di Masa Depan (Skala Asli)", xaxis_title="Waktu", yaxis_title="Nilai PM2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)
