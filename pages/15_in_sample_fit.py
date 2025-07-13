# pages/15_In_Sample_Fit.py
import streamlit as st
import pandas as pd
import numpy as np
import utils_for_made_model
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Kecocokan Model", page_icon="🎯", layout="wide")
st.title("🎯 Analisis Kecocokan Model (In-Sample Fit)")

st.info("""
Halaman ini tidak membuat peramalan masa depan. Sebaliknya, halaman ini menunjukkan seberapa baik **garis prediksi model menempel pada data historis yang digunakan untuk melatihnya**. 
Ini membantu kita memahami apakah model berhasil menangkap pola-pola utama dalam data.
""")

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

# --- ANTARMUKA PENGGUNA ---
st.header("Visualisasi In-Sample Fit", divider="blue")

# Opsi untuk memilih rentang waktu yang ingin dilihat
# Kita gunakan data asli untuk menentukan rentang tanggal yang tersedia
date_range = st.date_input(
    "Pilih rentang tanggal untuk dianalisis:",
    value=(st.session_state.main_dataframe_full.index.min().date(), st.session_state.main_dataframe_full.index.max().date()),
    min_value=st.session_state.main_dataframe_full.index.min().date(),
    max_value=st.session_state.main_dataframe_full.index.max().date(),
    help="Pilih rentang yang lebih sempit untuk melihat detail."
)

if len(date_range) == 2:
    start_date, end_date = date_range
    
    # --- PERBAIKAN LOGIKA DI SINI ---
    # 1. Dapatkan SEMUA prediksi in-sample langsung dari model (ini adalah Pandas Series)
    all_in_sample_predictions = final_model.fittedvalues
    
    # 2. Dapatkan data aktual yang digunakan untuk melatih model (ini adalah NumPy array)
    all_actual_data = final_model.model.endog

    # 3. Gabungkan keduanya dengan cara yang lebih robust
    #    Mulai dengan mengubah prediksi (yang sudah menjadi Pandas Series) menjadi DataFrame.
    plot_df = all_in_sample_predictions.to_frame(name='Prediksi In-Sample')
    #    Lalu tambahkan data aktual sebagai kolom baru. Pandas akan menyejajarkannya berdasarkan indeks.
    plot_df['Aktual'] = all_actual_data
    
    # 4. Potong (slice) DataFrame gabungan berdasarkan rentang tanggal dari pengguna
    filtered_plot_df = plot_df.loc[start_date:end_date]
    # --- AKHIR PERBAIKAN ---
    
    if filtered_plot_df.empty:
        st.warning("Tidak ada data yang tersedia untuk rentang tanggal yang dipilih.")
    else:
        # Buat grafik perbandingan dari data yang sudah difilter
        fig = utils_for_made_model.go.Figure()
        
        # Plot data aktual
        fig.add_trace(utils_for_made_model.go.Scatter(
            x=filtered_plot_df.index, 
            y=filtered_plot_df['Aktual'], 
            mode='lines', 
            name='Data PM2.5 Aktual (Asli)',
            line=dict(color='royalblue', width=2.5)
        ))
        
        # Plot prediksi in-sample
        fig.add_trace(utils_for_made_model.go.Scatter(
            x=filtered_plot_df.index, 
            y=filtered_plot_df['Prediksi In-Sample'], 
            mode='lines', 
            name='Kecocokan Model (In-Sample)',
            line=dict(color='red', dash='dash', width=1.5)
        ))
        
        fig.update_layout(
            title=f"Perbandingan Data Aktual vs. Kecocokan Model",
            xaxis_title="Waktu",
            yaxis_title="Nilai PM2.5 (µg/m³)"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Harap pilih rentang tanggal yang valid.")