import streamlit as st
import pandas as pd
import numpy as np
import utils_for_made_model  # Pastikan file utils_for_made_model.py Anda up-to-date
import os

def display_footer():
    """Menampilkan footer dengan informasi penulis dan institusi."""
    st.divider()
    st.markdown("""
    <div style="text-align: center; font-size: 0.9em; color: grey;">
        <p><b>Author:</b> Muhammad Kurnia Sani | <b>NIM:</b> 20.04.111.046</p>
        <p><b>Dosen Pembimbing:</b> (1) Dr. Rika Yunitarini, S.T., M.T. & (2) Kurniawan Eka Permana, S.Kom., M.Sc</p>
        <p><b>Dosen Penguji:</b> (1) Dr. Fika Hastarita Rachman, ST., M.Eng | (2) Moch. Kautsar Sophan, S.Kom., M.MT. | (3) Andharini Dwi Cahyani, S.Kom., M.Kom.,Ph.D</p>
        <p>Program Studi Teknik Informatika - Fakultas Teknik - Universitas Trunojoyo Madura</p>
    </div>
    """, unsafe_allow_html=True)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Otomatis", page_icon="🤖", layout="wide")
st.title("🤖 Prediksi Otomatis & Rekomendasi Kesehatan")

st.info("""
Halaman ini membuat peramalan PM2.5 dengan sebuah asumsi praktis: kondisi polutan dan cuaca di masa depan akan **berperilaku sama seperti beberapa jam terakhir yang tercatat**. 
Anda hanya perlu memilih periode waktu untuk melihat prediksi dan rekomendasi kesehatan yang dapat ditindaklanjuti.
""", icon="💡")

# --- PATH & PEMUATAN MODEL ---
MODEL_DIR = "models"
FULL_MODEL_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "full_prediction_artifacts.joblib")

artifacts = utils_for_made_model.load_prediction_artifacts(FULL_MODEL_ARTIFACTS_PATH)

if not artifacts:
    st.error("Model final (data penuh) belum dilatih. Harap latih model terlebih dahulu di halaman 'Prediksi (Data Penuh)'.")
    st.page_link("pages/13_prediction_full.py", label="Buka Halaman Pelatihan Model", icon="�")
    st.stop()

st.success(f"Model ARIMAX (dilatih dengan data penuh, order **{artifacts['final_order']}**) berhasil dimuat dan siap digunakan.", icon="✅")
final_model = artifacts['model']
scaler = artifacts['scaler']
final_features = artifacts['final_features']

if 'main_dataframe_full' not in st.session_state:
    st.session_state.main_dataframe_full = utils_for_made_model.load_station_data('Aotizhongxin')
    if st.session_state.main_dataframe_full is None:
        st.error("Gagal memuat data. Periksa path file di utils_for_made_model.py.")
        st.stop()

# --- ANTARMUKA PENGGUNA ---
st.header("Form Prediksi", divider="blue")

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
        exog_input_df = st.session_state.main_dataframe_full[final_features].tail(n_periods).copy()
        exog_input_df.reset_index(drop=True, inplace=True)

        with st.expander("Lihat Asumsi Kondisi Masa Depan (Berdasarkan Data Terakhir)"):
            st.write("Model menggunakan data dari jam-jam terakhir ini sebagai asumsi untuk kondisi di masa depan:")
            st.dataframe(exog_input_df)

        original_predictions, _ = utils_for_made_model.predict_future_values(
            final_model=final_model,
            scaler=scaler,
            exog_input_df=exog_input_df,
            final_features=final_features
        )
        
        last_timestamp = st.session_state.main_dataframe_full.index[-1]
        future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')
        
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
        
        # --- KARTU RINGKASAN BARU ---
        with st.container(border=True):
            # Ambil data untuk ringkasan
            last_known_pm25 = st.session_state.main_dataframe_full['PM2.5'].iloc[-1]
            next_hour_pred = result_df['Prediksi PM2.5 (µg/m³)'].iloc[0]
            overall_quality = result_df['Level Kualitas Udara'].mode()[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### KONDISI SAAT INI")
                st.metric("PM2.5 Terakhir Tercatat", f"{last_known_pm25:.2f} µg/m³")
            with col2:
                st.markdown("##### PREDIKSI JANGKA PENDEK")
                st.metric(f"Rata-rata nilai PM 2.5 untuk {selected_period} ke depan", f"{np.mean(result_df["Prediksi PM2.5 (µg/m³)"]):.2f} µg/m³")
            
            st.divider()
            st.markdown(f"**Ringkasan:** Kualitas udara secara umum untuk **{selected_period}** ke depan diprediksi akan berada pada level **{overall_quality}**.")
            st.markdown(f"Silahkan cek tabel di bawah ini untuk melihat **detail nilai PM 2.5 per-jam:**.")
        # --- AKHIR KARTU RINGKASAN ---

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

# Menampilkan footer di bagian bawah halaman
display_footer()
