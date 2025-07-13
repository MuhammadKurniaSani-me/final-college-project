# pages/3_data_preprocessing.py
import streamlit as st
import pandas as pd
import utils # Mengimpor file utilitas utama Anda

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Data Preprocessing", page_icon="⚙️", layout="wide")

st.title("⚙️ Proses Prapemrosesan Data")
st.markdown("""
Prapemrosesan adalah langkah krusial untuk membersihkan, mengubah, dan menyiapkan data mentah agar siap digunakan untuk pemodelan. 
Halaman ini mendemonstrasikan tiga langkah utama yang dilakukan: **Encoding**, **Imputasi Nilai Hilang**, dan **Normalisasi**.
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
if st.button("▶️ Jalankan Pipeline Preprocessing", type="primary", use_container_width=True):
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
        
    st.success("Pipeline Preprocessing Selesai!")

# --- Tampilan Hasil dengan TAB ---
if 'df_scaled' in st.session_state:
    tab1, tab2, tab3 = st.tabs(["**Langkah 1: Label Encoding**", "**Langkah 2: Imputasi Nilai Hilang**", "**Langkah 3: Normalisasi Min-Max**"])

    with tab1:
        st.header("Encoding Fitur Kategorikal", divider="blue")
        st.markdown("Mengubah fitur non-numerik seperti arah angin (`wd`) menjadi nilai numerik. Perhatikan kolom `wd` sebelum dan sesudah proses.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Encoding")
            st.dataframe(df_raw[['wd']].head())
        with col2:
            st.subheader("Sesudah Encoding")
            st.dataframe(st.session_state.df_encoded[['wd']].head())

    with tab2:
        st.header("Mengisi Nilai yang Hilang (Imputasi)", divider="blue")
        st.markdown("Menggunakan **interpolasi linear** untuk mengisi celah (nilai `NaN`) dalam data. Di bawah ini adalah contoh efek imputasi pada kolom `PM2.5`.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Imputasi (Contoh Data NaN)")
            # Tampilkan beberapa baris yang mengandung NaN pada PM2.5
            st.dataframe(st.session_state.df_encoded[st.session_state.df_encoded['PM2.5'].isnull()].head())
        with col2:
            st.subheader("Sesudah Imputasi")
            # Tampilkan baris yang sama setelah diimputasi
            st.dataframe(st.session_state.df_imputed.loc[st.session_state.df_encoded['PM2.5'].isnull().head().index])

    with tab3:
        st.header("Normalisasi Data dengan Min-Max Scaling", divider="blue")
        st.markdown("Menskalakan semua nilai fitur ke dalam rentang **[0, 1]**. Ini memastikan tidak ada satu fitur pun yang mendominasi model hanya karena skalanya lebih besar. Perhatikan bagaimana nilai pada kolom di bawah ini berubah dari skala aslinya menjadi nilai antara 0 dan 1.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Normalisasi (Data Asli)")
            # Tampilkan preview dari beberapa kolom data sebelum di-scaling
            st.dataframe(st.session_state.df_imputed[['PM10', 'CO', 'TEMP']].head())
        with col2:
            st.subheader("Sesudah Normalisasi (Skala 0-1)")
            # Tampilkan preview dari kolom yang sama setelah di-scaling
            st.dataframe(st.session_state.df_scaled[['PM10', 'CO', 'TEMP']].head())
