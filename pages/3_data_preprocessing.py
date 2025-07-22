# pages/3_data_preprocessing.py
import streamlit as st
import pandas as pd
import utils # Mengimpor file utilitas utama Anda

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Data _Preprocessing_", page_icon="⚙️", layout="wide")

st.title("⚙️ _Preprocessing_ Data")
st.markdown("""
_Preprocessing_ adalah langkah krusial untuk membersihkan, mengubah, dan menyiapkan data untuk performa sistem lebih baik. 
Halaman ini mendemonstrasikan tiga langkah utama yang dilakukan pada _preprocessing_: **Encoding**, **Imputation**, dan **Normalization**.
""")

# --- KEAMANAN & PEMUATAN DATA DARI SESSION STATE ---
if 'main_dataframe' not in st.session_state or st.session_state.main_dataframe is None:
    st.warning("🚨 Silakan pilih stasiun di halaman 'Ikhtisar Data' terlebih dahulu untuk memulai.")
    st.page_link("pages/2_data_overview.py", label="Kembali ke Halaman Ikhtisar Data", icon="📊")
    st.stop()

# Ambil data dari session state
df_raw = st.session_state.main_dataframe

st.info(f"Anda sedang bekerja dengan data dari stasiun: **{st.session_state.data_location}**")

# --- Tombol untuk Menjalankan Pipeline ---
if st.button("▶️ Jalankan Pipeline _Preprocessing_", type="primary", use_container_width=True):
    with st.spinner("Menjalankan proses..."):
        # Buat station code map
        station_names = [
            'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 
            'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'
        ]
        station_code_map = {name: i + 1 for i, name in enumerate(station_names)}

        # Langkah 1: Encoding
        df_encoded = utils.preprocess_encoding(df_raw, st.session_state.data_location, station_code_map)
        st.session_state.df_encoded = df_encoded
        
        # Langkah 2: Imputasi
        df_imputed = utils.preprocess_impute(df_encoded)
        st.session_state.df_imputed = df_imputed
        
        # Langkah 3: Normalisasi
        df_scaled, scaler = utils.preprocess_scale(df_imputed)
        st.session_state.df_scaled = df_scaled
        st.session_state.scaler = scaler # Simpan scaler untuk digunakan di halaman prediksi
        
    st.success("Pipeline _Preprocessing_ Selesai!")

# --- Tampilan Hasil dengan TAB ---
if 'df_scaled' in st.session_state:
    tab1, tab2, tab3 = st.tabs(["**Langkah 1: Label Encoding**", "**Langkah 2: Missing Values Imputation**", "**Langkah 3: Min-Max Normalization**"])

    with tab1:
        st.header("Encoding _feature_ (kolom tabel) Kategorikal (data non-numerik)", divider="blue")
        st.markdown("Mengubah fitur non-numerik seperti arah angin (`wd`) menjadi nilai numerik. Perhatikan kolom `wd` sebelum dan sesudah proses.")
        st.page_link("https://doi.org/10.1186/s40537-021-00548-1", label="Bekkar A. et al. (2021)", icon="🔗")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum _encoding_")
            st.dataframe(df_raw[['wd']].head())
        with col2:
            st.subheader("Sesudah _encoding_")
            st.dataframe(st.session_state.df_encoded[['wd']].head())

    with tab2:
        st.header("Mengisi Nilai yang Hilang (_Missing Values Imputation_)", divider="blue")
        st.markdown("Menggunakan metode **interpolasi linear** untuk mengisi _missing value_ (`NaN`) dalam data. Di bawah ini adalah proses _imputation_ pada kolom `PM2.5`.")
        st.page_link("https://doi.org/10.1186/s40537-021-00548-1", label="Bekkar A. et al. (2021)", icon="🔗")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum _imputation_")
            # Tampilkan beberapa baris yang mengandung NaN pada PM2.5
            st.dataframe(st.session_state.df_encoded[st.session_state.df_encoded['PM2.5'].isnull()].head())
        with col2:
            st.subheader("Sesudah _imputation_")
            # Tampilkan baris yang sama setelah diimputasi
            st.dataframe(st.session_state.df_imputed.loc[st.session_state.df_encoded['PM2.5'].isnull().head().index])

    with tab3:
        st.header("_Min-Max Normalization_", divider="blue")
        st.markdown("Menyeimbangkan skala semua nilai ke dalam rentang **[0, 1]**. Ini memastikan tidak ada satu fitur pun yang mendominasi model hanya karena skalanya lebih besar. Perhatikan bagaimana nilai pada kolom di bawah ini berubah sebelum dan sesudah _Min-Max normalization_.")
        st.page_link("https://doi.org/10.1186/s40537-021-00548-1", label="Bekkar A. et al. (2021)", icon="🔗")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum _normalization_")
            # Tampilkan preview dari beberapa kolom data sebelum di-scaling
            st.dataframe(st.session_state.df_imputed[['PM10', 'CO', 'TEMP']].head())
        with col2:
            st.subheader("Sesudah _normalization_")
            # Tampilkan preview dari kolom yang sama setelah di-scaling
            st.dataframe(st.session_state.df_scaled[['PM10', 'CO', 'TEMP']].head())
