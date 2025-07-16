# pages/14_Auto_Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import utils_for_made_model  # Pastikan file utils_for_made_model.py Anda up-to-date
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Otomatis", page_icon="🤖", layout="wide")
st.title("🤖 Prediksi Kualitas Udara & Rekomendasi Kesehatan")

st.info("""
Halaman ini membuat peramalan PM2.5 dengan asumsi kondisi masa depan akan sama seperti beberapa jam terakhir yang tercatat dalam dataset. 
Pilih periode waktu untuk melihat prediksi dan rekomendasi kesehatan yang dapat ditindaklanjuti.
""", icon="💡")

# --- PATH & PEMUATAN MODEL ---
MODEL_DIR = "models"
FULL_MODEL_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "full_prediction_artifacts.joblib")

# Muat artefak model. Fungsi ini di-cache untuk kecepatan.
artifacts = utils_for_made_model.load_prediction_artifacts(FULL_MODEL_ARTIFACTS_PATH)

# Periksa apakah model sudah ada. Jika tidak, hentikan halaman.
if not artifacts:
    st.error("Model final (data penuh) belum dilatih. Harap latih model terlebih dahulu di halaman 'Prediksi (Data Penuh)'.")
    st.page_link("pages/13_prediction_full.py", label="Buka Halaman Pelatihan Model", icon="🔮")
    st.stop()

# Jika model ada, muat semua komponennya
st.success(f"Model final (data penuh) dengan order **{artifacts['final_order']}** berhasil dimuat.", icon="✅")
final_model = artifacts['model']
scaler = artifacts['scaler']
final_features = artifacts['final_features']

# Muat data penuh untuk digunakan sebagai sumber nilai eksogen
if 'main_dataframe_full' not in st.session_state:
    st.session_state.main_dataframe_full = utils_for_made_model.load_full_station_data('Aotizhongxin')
    if st.session_state.main_dataframe_full is None:
        st.error("Gagal memuat data. Periksa path file di utils_for_made_model.py.")
        st.stop()

# --- ANTARMUKA PENGGUNA ---
st.header("Form Prediksi", divider="blue")

# Definisikan opsi periode dalam dictionary
period_options = {
    "1 Hari (24 Jam)": 24,
    "3 Hari (72 Jam)": 72,
    "1 Minggu (168 Jam)": 168,
    "2 Minggu (336 Jam)": 336,
    "1 Bulan (720 Jam)": 720
}

selected_period = st.selectbox(
    "Pilih Periode Prediksi Masa Depan:",
    options=list(period_options.keys()),
    help="Jumlah jam ini juga akan menentukan data historis mana yang digunakan sebagai asumsi nilai eksogen."
)

if st.button("Buat Prediksi & Rekomendasi", type="primary", use_container_width=True):
    n_periods = period_options[selected_period]
    
    with st.spinner(f"Membuat peramalan untuk {selected_period}..."):
        # Ambil 'n_periods' baris terakhir dari data historis sebagai input masa depan
        exog_input_df = st.session_state.main_dataframe_full[final_features].tail(n_periods).copy()
        exog_input_df.reset_index(drop=True, inplace=True)

        with st.expander("Lihat Asumsi Nilai Eksogen yang Digunakan"):
            st.write("Model menggunakan data dari jam-jam terakhir ini sebagai asumsi untuk kondisi di masa depan:")
            st.dataframe(exog_input_df)

        # Panggil fungsi prediksi
        original_predictions, scaled_predictions = utils_for_made_model.predict_future_values(
            final_model=final_model,
            scaler=scaler,
            exog_input_df=exog_input_df,
            final_features=final_features
        )
        
        # Siapkan indeks waktu masa depan
        last_timestamp = st.session_state.main_dataframe_full.index[-1]
        future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')
        
        # Buat tabel hasil yang kaya informasi
        results_data = []
        for i, pred_value in enumerate(original_predictions):
            level, recommendation, _ = utils_for_made_model.get_air_quality_advice(pred_value)
            results_data.append({
                "Waktu Prediksi": future_index[i],
                "Prediksi PM2.5 (µg/m³)": pred_value,
                "Level Kualitas Udara": level,
                "Rekomendasi": recommendation
            })
        
        result_df = pd.DataFrame(results_data)
        
        # --- Tampilan Hasil ---
        st.subheader("Hasil Peramalan & Rekomendasi")
        
        # Tampilkan ringkasan kualitas udara
        overall_quality = result_df['Level Kualitas Udara'].mode()[0]
        st.metric("Kualitas Udara Umumnya Akan:", overall_quality)

        # Tampilkan tabel hasil dengan format kolom yang lebih baik
        st.dataframe(result_df, use_container_width=True,
                     column_config={
                         "Waktu Prediksi": st.column_config.DatetimeColumn("Waktu", format="YYYY-MM-DD HH:mm"),
                         "Prediksi PM2.5 (µg/m³)": st.column_config.NumberColumn(
                             "Prediksi PM2.5 (µg/m³)",
                             format="%.2f"
                         ),
                         "Rekomendasi": st.column_config.TextColumn("Rekomendasi", width="large")
                     })

        # Tampilkan grafik
        st.subheader("Grafik Prediksi Kontekstual")
        historical_data_unscaled = st.session_state.main_dataframe_full['PM2.5'].tail(n_periods * 2)
        fig = utils_for_made_model.go.Figure()
        fig.add_trace(utils_for_made_model.go.Scatter(x=historical_data_unscaled.index, y=historical_data_unscaled, mode='lines', name='Histori PM2.5 (Skala Asli)'))
        fig.add_trace(utils_for_made_model.go.Scatter(x=result_df["Waktu Prediksi"], y=result_df["Prediksi PM2.5 (µg/m³)"], mode='lines+markers', name='Prediksi PM2.5 (Skala Asli)', line=dict(color='red')))
        fig.update_layout(title="Visualisasi Prediksi di Masa Depan (Skala Asli)", xaxis_title="Waktu", yaxis_title="Nilai PM2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)
