# pages/7_prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import utils
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Baru", page_icon="🔮", layout="wide")
st.title("🔮 Halaman Prediksi PM2.5")

# --- PATH UNTUK MENYIMPAN ARTEFAK ---
MODEL_DIR = "models"
ARTIFACTS_PATH = os.path.join(MODEL_DIR, "prediction_artifacts.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "final_arimax_model.pkl")

# --- PERBAIKAN UTAMA DI SINI: MEMBUAT HALAMAN MANDIRI ---
# Cek apakah data utama ada di sesi. Jika tidak, muat data default.
if 'main_dataframe' not in st.session_state:
    st.info("Memuat data default (Aotizhongxin) untuk sesi ini...", icon="ℹ️")
    # Muat data dari stasiun default untuk mengisi nilai awal form
    st.session_state.main_dataframe = utils.load_station_data('Aotizhongxin')

# Jika data gagal dimuat, hentikan aplikasi
if st.session_state.main_dataframe is None:
    st.error("Gagal memuat data default. Periksa path file di `utils.py`.")
    st.stop()
# --- AKHIR PERBAIKAN ---

# --- KONTROL UTAMA: LOAD MODEL ATAU LATIH MODEL ---
# Coba muat artefak yang sudah ada
artifacts = utils.load_prediction_artifacts(ARTIFACTS_PATH)

if artifacts:
    # JIKA MODEL SUDAH ADA, LANGSUNG KE TAHAP PREDIKSI
    st.success(f"Model final ARIMAX (order **{artifacts['final_order']}**) berhasil dimuat dari file.", icon="✅")
    
    final_model = artifacts['model']
    scaler = artifacts['scaler']
    final_features = artifacts['final_features']
    original_cols = artifacts['original_cols']

    st.header("Input Data untuk Prediksi", divider="blue")
    n_periods = st.number_input("Berapa jam ke depan yang ingin Anda prediksi?", min_value=1, max_value=24, value=4)
    with st.form("prediction_form"):
        st.markdown(f"Masukkan nilai **(dalam skala asli)** untuk **{len(final_features)} fitur** berikut untuk setiap jamnya.")
        last_known_values = st.session_state.main_dataframe[final_features].iloc[-1]
        input_data = {feature: [last_known_values[feature]] * n_periods for feature in final_features}
        edited_df = st.data_editor(pd.DataFrame(input_data), num_rows="dynamic", use_container_width=True)
        submitted = st.form_submit_button("Buat Prediksi", type="primary", use_container_width=True)
        
        # --- PROSES & TAMPILAN PREDIKSI ---
        if submitted:
            input_df = pd.DataFrame(edited_df)
            if input_df.isnull().values.any():
                st.error("⚠️ GAGAL: Terdapat sel kosong pada tabel input. Harap isi semua nilai.")
            else:
                with st.spinner("Membuat peramalan..."):
                    exog_input = input_df.astype(float)
                    
                    # original_predictions, scaled_predictions = utils.predict_future_values(
                    #     final_model=final_model, scaler=scaler, exog_input_df=exog_input,
                    #     final_features=final_features, original_cols_for_scaler=final_features
                    # )

                    original_predictions, scaled_predictions = utils.predict_future_values(
                        final_model=final_model,
                        scaler=scaler,
                        exog_input_df=exog_input,
                        final_features=final_features
                    )

                    # --- PERBAIKAN LOGIKA FINAL & SEDERHANA ---
                    # Dapatkan timestamp terakhir langsung dari indeks DataFrame yang sudah benar
                    last_timestamp = st.session_state.main_dataframe.index[-1]
                    
                    # Buat rentang tanggal masa depan dari timestamp terakhir
                    future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')
                    # --- AKHIR PERBAIKAN ---
                    
                    # Tampilkan tabel hasil
                    result_df = pd.DataFrame({"Waktu Prediksi": future_index, "Prediksi PM2.5 (µg/m³)": original_predictions})
                    st.subheader("Hasil Peramalan")
                    st.dataframe(result_df.style.format({"Waktu Prediksi": "{:%Y-%m-%d %H:%M}", "Prediksi PM2.5 (µg/m³)": "{:.2f}"}), use_container_width=True)

                # --- LOGIKA BARU UNTUK VISUALISASI SKALA ASLI ---
                    st.subheader("Grafik Prediksi Kontekstual")
                    
                    historical_window_size = n_periods * n_periods
                    st.markdown(f"Untuk memprediksi **{n_periods} jam** ke depan, kami menampilkan **{historical_window_size} jam** data historis sebagai konteks.")

                    # 1. Ambil data historis dari DataFrame utama (SKALA ASLI)
                    # Kita gunakan df_imputed untuk memastikan tidak ada NaN di plot
                    historical_data_unscaled = st.session_state.main_dataframe['PM2.5'].tail(historical_window_size)
                    
                    # 2. Reset indeks untuk mendapatkan sumbu-X numerik
                    historical_data_for_plot = historical_data_unscaled.reset_index(drop=True)
                    
                    # 3. Buat indeks numerik untuk prediksi
                    last_numerical_index = historical_data_for_plot.index[-1]
                    future_numerical_index = np.arange(last_numerical_index + 1, last_numerical_index + 1 + n_periods)
                    
                    # 5. Buat grafik menggunakan indeks numerik yang konsisten
                    fig = utils.go.Figure()
                    
                    fig.add_trace(utils.go.Scatter(
                        x=historical_data_for_plot.index, 
                        y=historical_data_for_plot.values, 
                        mode='lines', name='Data Historis (Ternormalisasi)'
                    ))
                    
                    fig.add_trace(utils.go.Scatter(
                        x=future_numerical_index, 
                        y=original_predictions, # Gunakan prediksi yang ternormalisasi untuk plot ini
                        mode='lines+markers', name='Hasil Prediksi (Ternormalisasi)', 
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Visualisasi Prediksi di Masa Depan (Skala Ternormalisasi)",
                        xaxis_title="Indeks Waktu (Urutan Data)", # Judul sumbu-X diubah
                        yaxis_title="Nilai PM2.5 (Ternormalisasi)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # --- AKHIR PERBAIKAN ---
else:
    # --- PERUBAHAN LOGIKA UTAMA DI SINI ---
    # JIKA MODEL BELUM ADA, TAMPILKAN OPSI UNTUK MELATIH BERDASARKAN SKENARIO TERBAIK
    st.warning("Model final belum ada. Anda perlu melatih dan menyimpannya sekali.", icon="⚠️")
    st.info("""
    Klik tombol di bawah untuk melatih model final berdasarkan **Skenario 4 (Seleksi Fitur & Penghapusan Outlier)**, 
    yang telah terbukti menjadi yang terbaik pada halaman evaluasi.
    """)

    if st.button("Latih dan Simpan Model Final (Skenario 4)", type="primary"):
        # Pastikan data yang dibutuhkan ada di session state
        if 'df_imputed' not in st.session_state:
            st.error("Data yang telah diproses tidak ditemukan. Harap jalankan halaman Preprocessing terlebih dahulu.")
            st.stop()

        # Pastikan folder 'models' ada
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        with st.spinner("Melatih model final... Ini mungkin memakan waktu beberapa menit."):
            # Hardcode parameter untuk Skenario 4
            fs_params = {'corr_threshold': 0.5, 'vif_threshold': 10.0}
            or_params = {'k_optimal': 2, 'sil_threshold': 0.5}
            
            station_names = st.session_state.get('locations_name', [st.session_state.data_location])
            station_code_map = {name: i + 1 for i, name in enumerate(station_names)}
            full_df = utils.load_station_data(st.session_state.data_location, num_rows=False)
            # Langkah 1: Encoding
            full_df_encoded = utils.preprocess_encoding(full_df, st.session_state.data_location, station_code_map)
            # Langkah 2: Imputasi
            full_df_imputed = utils.preprocess_impute(full_df_encoded)
            # Langkah 3: Normalisasi
            full_df_scaled, st.session_state.scaler = utils.preprocess_scale(full_df_imputed)
            # st.dataframe(full_df_scaled)

            trained_model, fitted_scaler, features, original_cols_list, order = utils.train_final_model_from_best_scenario(
                full_df_imputed=full_df_imputed,
                use_fs=True,  # Sesuai Skenario 4
                use_or=True,  # Sesuai Skenario 4
                fs_params=fs_params, 
                or_params=or_params
            )

            # Simpan model statsmodels
            trained_model.save(MODEL_PATH)
            
            # Siapkan dan simpan dictionary artefak
            artifacts_to_save = {
                'model_path': MODEL_PATH, 'scaler': fitted_scaler,
                'final_features': features, 'original_cols': original_cols_list,
                'final_order': order
            }
            joblib.dump(artifacts_to_save, ARTIFACTS_PATH)
        
        st.success(f"Model dan artefak berhasil disimpan di `{ARTIFACTS_PATH}`. Silakan refresh halaman untuk mulai membuat prediksi.")
        st.balloons()