# pages/13_prediction_full.py
import streamlit as st
import pandas as pd
import numpy as np
import utils_for_made_model
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi (Data Penuh)", page_icon="🔮", layout="wide")
st.title("🔮 Halaman Prediksi (Menggunakan Data Penuh)")

# --- PATH UNTUK MENYIMPAN ARTEFAK ---
# Kita gunakan nama file model yang berbeda agar tidak tertimpa
MODEL_DIR = "models"
FULL_MODEL_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "full_prediction_artifacts.joblib")
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "full_final_arimax_model.pkl")
station_name = 'Changping'

st.info("""
Halaman ini menggunakan **seluruh dataset historis (35.000+ baris)** untuk melatih model final. 
Proses pelatihan pertama kali akan **sangat lama**, namun model akan disimpan agar proses selanjutnya menjadi instan.
""", icon="ℹ️")

# --- KONTROL UTAMA: LOAD MODEL ATAU LATIH MODEL ---
artifacts = utils_for_made_model.load_prediction_artifacts(FULL_MODEL_ARTIFACTS_PATH)

if artifacts:
    # JIKA MODEL SUDAH ADA, LANGSUNG KE TAHAP PREDIKSI
    st.success(f"Model final (data penuh) dengan order **{artifacts['final_order']}** berhasil dimuat dari file.", icon="✅")
    
    final_model = artifacts['model']
    scaler = artifacts['scaler']
    final_features = artifacts['final_features']

    with st.expander("🔬 Klik untuk Melihat Persamaan Matematis Model"):
        st.markdown("""
        Tabel di bawah ini adalah ringkasan dari model ARIMAX yang telah dilatih. **Kolom `coef`** adalah bagian terpenting; ini menunjukkan koefisien atau "bobot" yang telah dipelajari model untuk setiap variabel.
        
        Secara konseptual, persamaan prediksinya adalah:
        
        `Prediksi PM2.5 = (coef_NO2 * Nilai_NO2) + (coef_PM10 * Nilai_PM10) + ... + (coef_ar.L1 * PM2.5_sebelumnya) + ...`
        
        Ini adalah representasi matematis dari "pengetahuan" model.
        """)
        st.text(final_model.summary())

    # Muat data penuh untuk nilai default form
    if 'main_dataframe_full' not in st.session_state:
        st.session_state.main_dataframe_full = utils_for_made_model.load_station_data(station_name)

    # --- ANTARMUKA INPUT PENGGUNA ---
    st.header("Input Data untuk Prediksi", divider="blue")
    n_periods = st.number_input("Berapa jam ke depan yang ingin Anda prediksi?", min_value=1, max_value=24, value=4, key="full_n_periods")
    
    with st.form("full_prediction_form"):
        st.markdown(f"Masukkan nilai **(dalam skala asli)** untuk **{len(final_features)} fitur** berikut untuk setiap jamnya.")
        last_known_values = st.session_state.main_dataframe_full[final_features].iloc[-1]
        input_data = {feature: [last_known_values[feature]] * n_periods for feature in final_features}
        edited_df = st.data_editor(pd.DataFrame(input_data), num_rows="dynamic", use_container_width=True, key="full_data_editor")
        submitted = st.form_submit_button("Buat Prediksi", type="primary", use_container_width=True)

    if submitted:
        # Logika prediksi dan tampilan hasil
        input_df_validated = edited_df.astype(float)
        original_predictions, _ = utils_for_made_model.predict_future_values(
            final_model=final_model, scaler=scaler, exog_input_df=input_df_validated,
            final_features=final_features
        )
        
        # Dapatkan timestamp terakhir langsung dari indeks DataFrame yang sudah benar
        last_timestamp = st.session_state.main_dataframe_full.index[-1]
        
        # Buat rentang tanggal masa depan dari timestamp terakhir
        future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')
        # --- AKHIR PERBAIKAN ---
        
        # Tampilkan tabel hasil
        result_df = pd.DataFrame({"Waktu Prediksi": future_index, "Prediksi PM2.5 (µg/m³)": original_predictions})
        st.subheader("Hasil Peramalan")
        st.dataframe(result_df.style.format({"Waktu Prediksi": "{:%Y-%m-%d %H:%M}", "Prediksi PM2.5 (µg/m³)": "{:.2f}"}), use_container_width=True)

        # --- LOGIKA BARU UNTUK VISUALISASI LENGKAP ---
        st.subheader("Grafik Prediksi Kontekstual")

        historical_window_size = n_periods * n_periods
        st.markdown(f"Untuk memprediksi **{n_periods} jam** ke depan, kami menampilkan **{historical_window_size} jam** data historis sebagai konteks.")

        # 1. Siapkan semua data yang diperlukan (historis dan masa depan)
        # Data historis dalam skala asli
        hist_pm25 = st.session_state.main_dataframe_full['PM2.5'].tail(historical_window_size)
        hist_pm10 = st.session_state.main_dataframe_full['PM10'].tail(historical_window_size)
        hist_so2 = st.session_state.main_dataframe_full['SO2'].tail(historical_window_size)
        hist_no2 = st.session_state.main_dataframe_full['NO2'].tail(historical_window_size)
        hist_co = st.session_state.main_dataframe_full['CO'].tail(historical_window_size)

        # Data masa depan dari input pengguna
        future_pm10 = edited_df['PM10']
        future_so2 = edited_df['SO2']
        future_no2 = edited_df['NO2']
        future_co = edited_df['CO']

        # Indeks waktu masa depan
        last_timestamp = st.session_state.main_dataframe_full.index[-1]
        future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')

        # 2. Buat grafik dengan dua sumbu Y
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # --- TAMBAHKAN TRACE UNTUK SUMBU Y PRIMER ---
        # PM2.5 (Paling Penting)
        fig.add_trace(utils_for_made_model.go.Scatter(x=hist_pm25.index, y=hist_pm25, mode='lines', 
                                name='Histori PM2.5', line=dict(color='royalblue', width=3)), secondary_y=False)
        fig.add_trace(utils_for_made_model.go.Scatter(x=future_index, y=original_predictions, mode='lines+markers', 
                                name='Prediksi PM2.5', line=dict(color='red', dash='dash', width=3)), secondary_y=False)

        # PM10 (Fitur Pendukung)
        fig.add_trace(utils_for_made_model.go.Scatter(x=hist_pm10.index, y=hist_pm10, mode='lines', 
                                name='Histori PM10', line=dict(color='lightgreen', width=1.5)), secondary_y=False)
        fig.add_trace(utils_for_made_model.go.Scatter(x=future_index, y=future_pm10, mode='lines+markers', 
                                name='Input PM10', line=dict(color='green', dash='dot', width=2)), secondary_y=False)

        # SO2 (Fitur Pendukung)
        fig.add_trace(utils_for_made_model.go.Scatter(x=hist_so2.index, y=hist_so2, mode='lines', 
                                name='Histori SO2', line=dict(color='lightblue', width=1.5)), secondary_y=False)
        fig.add_trace(utils_for_made_model.go.Scatter(x=future_index, y=future_so2, mode='lines+markers', 
                                name='Input SO2', line=dict(color='cyan', dash='dot', width=2)), secondary_y=False)

        # NO2 (Fitur Pendukung)
        fig.add_trace(utils_for_made_model.go.Scatter(x=hist_no2.index, y=hist_no2, mode='lines', 
                                name='Histori NO2', line=dict(color='lightgray', width=1.5)), secondary_y=False)
        fig.add_trace(utils_for_made_model.go.Scatter(x=future_index, y=future_no2, mode='lines+markers', 
                                name='Input NO2', line=dict(color='orange', dash='dot', width=2)), secondary_y=False)

        # --- TAMBAHKAN TRACE UNTUK CO (Sumbu Y Sekunder) ---
        fig.add_trace(utils_for_made_model.go.Scatter(x=hist_co.index, y=hist_co, mode='lines', 
                                name='Histori CO', line=dict(color='lightpink', width=1.5)), secondary_y=True)
        fig.add_trace(utils_for_made_model.go.Scatter(x=future_index, y=future_co, mode='lines+markers', 
                                name='Input CO', line=dict(color='purple', dash='dot', width=2)), secondary_y=True)


        # --- Update Layout Grafik ---
        fig.update_layout(
            title_text="Visualisasi Prediksi PM2.5 vs. Skenario Input Lengkap",
            xaxis_title="Waktu",
            legend_title="Legenda"
        )
        # Atur judul sumbu Y
        fig.update_yaxes(title_text="<b>Nilai Polutan Primer</b> (µg/m³)", secondary_y=False)
        fig.update_yaxes(title_text="<b>Nilai CO</b>", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
        # --- AKHIR LOGIKA BARU ---

else:
    # JIKA MODEL BELUM ADA, TAMPILKAN OPSI UNTUK MELATIH
    st.warning("Model final (data penuh) belum ada. Anda perlu melatih dan menyimpannya sekali.", icon="⚠️")
    
    if st.button("Latih dan Simpan Model Final (Data Penuh)", type="primary"):
        # Pastikan folder 'models' ada
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        with st.spinner("Memuat seluruh data dan melatih model final... Proses ini akan SANGAT LAMA."):
            # Muat seluruh data dari stasiun default
            full_df = utils_for_made_model.load_station_data(station_name)
            if full_df is not None:
                # Latih model final menggunakan Skenario 4 sebagai default
                fs_params = {'corr_threshold': 0.3, 'vif_threshold': 10.0}
                or_params = {'k_optimal': 2, 'sil_threshold': 0.3}
                
                trained_model, fitted_scaler, features, original_cols_list, order = utils_for_made_model.train_final_model_from_best_scenario(
                    full_df=full_df, # Gunakan data penuh
                    use_fs=True, use_or=True, # Asumsi Skenario 4
                    fs_params=fs_params, or_params=or_params,
                    station_name=station_name
                )

                # Simpan model dan artefak
                trained_model.save(FULL_MODEL_PATH)
                artifacts_to_save = {
                    'model_path': FULL_MODEL_PATH, 'scaler': fitted_scaler,
                    'final_features': features, 'original_cols': original_cols_list,
                    'final_order': order
                }
                joblib.dump(artifacts_to_save, FULL_MODEL_ARTIFACTS_PATH)
                st.success(f"Model data penuh berhasil disimpan. Silakan refresh halaman.")
                st.balloons()
            else:
                st.error("Gagal memuat data penuh untuk pelatihan.")
