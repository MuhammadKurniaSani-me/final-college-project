# pages/7_prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import utils

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Baru", page_icon="🔮", layout="wide")

st.title("🔮 Halaman Prediksi PM2.5")
st.markdown("""
Halaman ini memungkinkan Anda untuk membuat peramalan PM2.5 di masa depan. 
Model ARIMAX final dilatih pada keseluruhan data dari skenario terbaik yang ditemukan di halaman evaluasi.
""")

# --- KEAMANAN & PEMUATAN DATA ---
required_states = ['df_cleaned', 'final_features', 'df_imputed', 'scaler', 'main_dataframe']
if not all(state in st.session_state for state in required_states):
    st.warning("🚨 Pastikan semua pipeline di halaman sebelumnya telah dijalankan.")
    st.stop()

# --- TENTUKAN SKENARIO TERBAIK & LATIH MODEL FINAL ---
with st.container(border=True):
    df_cleaned = st.session_state.df_scaled
    # final_features = st.session_state.final_features
    scaler = st.session_state.scaler
    all_feature_names = st.session_state.df_imputed.columns.tolist()

    # 1. Tentukan skenario terbaik dari hasil evaluasi
    eval_results = st.session_state.eval_results
    best_scenario_name = min(eval_results, key=lambda x: eval_results[x]['avg_rmse'])
    best_scenario_params = eval_results[best_scenario_name]

    st.success(f"Skenario terbaik yang dipilih adalah **'{best_scenario_name}'** dengan rata-rata RMSE **{best_scenario_params['avg_rmse']:.4f}**.")

    fs_params = {'corr_threshold': 0.3, 'vif_threshold': 10.0} # Ganti jika Anda membuatnya dinamis
    or_params = {'k_optimal': 2, 'sil_threshold': 0.0}

    final_model, scaler, final_features, original_cols, final_order = utils.train_final_model_from_best_scenario(
        full_df=df_cleaned,
        use_fs='Seleksi Fitur' in best_scenario_name,
        use_or='Penghapusan Outlier' in best_scenario_name,
        fs_params=fs_params,
        or_params=or_params
        )
    st.info(f"Model final berhasil dilatih dengan order **ARIMAX{final_order}** menggunakan **{len(final_features)}** fitur eksogen.")
st.success(f"Model final ARIMAX telah siap digunakan dengan order terbaik: **{final_order}**", icon="✅")

# --- TAMBAHKAN BLOK DEBUGGING INI ---
with st.expander("🔬 Klik untuk Melihat Ringkasan Model Final"):
    st.text(final_model.summary())
# --- AKHIR BLOK DEBUGGING ---


# --- ANTARMUKA INPUT PENGGUNA ---
st.header("Input Data untuk Prediksi", divider="blue")
n_periods = st.number_input("Berapa jam ke depan yang ingin Anda prediksi?", min_value=1, max_value=24, value=4)

with st.form("prediction_form"):
    st.markdown(f"Masukkan nilai **(dalam skala asli)** untuk **{len(final_features)} fitur** berikut untuk setiap jamnya.")
    last_known_values = st.session_state.main_dataframe[final_features].iloc[-1]
    input_data = {feature: [last_known_values[feature]] * n_periods for feature in final_features}
    edited_df = st.data_editor(pd.DataFrame(input_data), num_rows="dynamic", use_container_width=True)
    submitted = st.form_submit_button("Buat Prediksi", type="primary", use_container_width=True)

# ... (kode di atasnya, termasuk 'with st.form(...):', tetap sama) ...

# --- PROSES & TAMPILAN PREDIKSI (VERSI BARU DENGAN PLOT DINAMIS) ---
if submitted:
    # Buat DataFrame dari input pengguna untuk validasi
    input_df = pd.DataFrame(edited_df)

    # --- PERBAIKAN UTAMA DI SINI: PASTIKAN TIPE DATA BENAR ---
    try:
        # Coba konversi semua input menjadi tipe float. Ini sangat penting.
        exog_input = input_df.astype(float)
    except ValueError:
        st.error("⚠️ GAGAL: Pastikan semua input pada tabel adalah angka yang valid.")
        st.stop() # Hentikan eksekusi jika ada input non-numerik
    # --- AKHIR PERBAIKAN ---

    # Validasi input sebelum melanjutkan
    if input_df.isnull().values.any():
        st.error("⚠️ GAGAL: Terdapat sel kosong pada tabel input. Harap isi semua nilai.")
    else:
        with st.spinner("Membuat peramalan..."):
            # Lanjutkan dengan exog_input yang sudah divalidasi
            exog_input = input_df
            
            original_predictions, scaled_predictions = utils.predict_future_values(
                final_model=final_model,
                scaler=scaler,
                exog_input_df=exog_input,
                final_features=final_features,
                original_cols=all_feature_names
            )


            # Buat indeks waktu masa depan yang sebenarnya
            last_row = st.session_state.main_dataframe.iloc[-1]
            last_timestamp_dict = last_row[['year', 'month', 'day', 'hour']].to_dict()
            last_timestamp_df = pd.DataFrame(last_timestamp_dict, index=[0])
            last_timestamp = pd.to_datetime(last_timestamp_df)[0]
            future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_periods, freq='h')
            
            # Tampilkan tabel hasil (dalam skala asli)
            result_df = pd.DataFrame({"Waktu Prediksi": future_index, "Prediksi PM2.5 (µg/m³)": original_predictions})
            st.subheader("Hasil Peramalan")
            st.dataframe(result_df.style.format({"Waktu Prediksi": "{:%Y-%m-%d %H:%M}", "Prediksi PM2.5 (µg/m³)": "{:.2f}"}), use_container_width=True)

            # --- LOGIKA BARU UNTUK VISUALISASI DINAMIS ---
            st.subheader("Grafik Prediksi Kontekstual")
            
            # 1. Tentukan ukuran jendela historis berdasarkan permintaan Anda (n^2)
            historical_window_size = n_periods * n_periods
            st.markdown(f"Untuk memprediksi **{n_periods} jam** ke depan, kami menampilkan **{historical_window_size} jam** data historis sebagai konteks.")

            # 2. Ambil data historis dari DataFrame asli (skala asli)
            # Kita gunakan data yang sudah diimputasi agar tidak ada celah di grafik
            historical_data = st.session_state.df_imputed['PM2.5'].tail(historical_window_size)
            
            # 3. Buat grafik dengan data skala asli
            fig = utils.go.Figure()
            
            # Plot data historis dalam skala asli
            fig.add_trace(utils.go.Scatter(
                x=historical_data.index, 
                y=historical_data, 
                mode='lines', 
                name='Data Historis (Skala Asli)'
            ))
            
            # Plot data prediksi dalam skala asli
            fig.add_trace(utils.go.Scatter(
                x=future_index, 
                y=original_predictions, 
                mode='lines+markers', 
                name='Hasil Prediksi (Skala Asli)', 
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Visualisasi Prediksi di Masa Depan (Skala Asli)",
                xaxis_title="Waktu", 
                yaxis_title="Nilai PM2.5 (µg/m³)" # Sumbu Y sekarang dalam skala asli
            )
            st.plotly_chart(fig, use_container_width=True)
            # --- AKHIR LOGIKA BARU ---