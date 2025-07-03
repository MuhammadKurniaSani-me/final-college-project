# pages/3_data_preprocessing.py
import streamlit as st
import pandas as pd
import utils # Mengimpor file utilitas kita

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Data Preprocessing", page_icon="⚙️", layout="wide")

st.title("⚙️ Proses Prapemrosesan Data")
st.markdown("""
Prapemrosesan adalah langkah krusial untuk membersihkan, mengubah, dan menyiapkan data mentah agar siap digunakan untuk pemodelan. 
Halaman ini mendemonstrasikan tiga langkah utama yang dilakukan: **Encoding**, **Imputasi Nilai Hilang**, dan **Normalisasi**.
""")

# --- KEAMANAN & PEMUATAN DATA DARI SESSION STATE ---
if 'main_dataframe' not in st.session_state:
    st.warning("🚨 Silakan pilih stasiun di halaman 'Ikhtisar Data' terlebih dahulu untuk memulai.")
    st.stop()

# Ambil data dari session state
df_raw = st.session_state.main_dataframe.copy()
# Ambil sampel data untuk demonstrasi yang lebih cepat
df_sample = df_raw.copy()

# --- PERUBAHAN DI SINI: DEFINISIKAN DAN HAPUS FITUR WAKTU ---
time_features = ['year', 'month', 'day', 'hour']
# Buat DataFrame baru yang hanya berisi fitur untuk diproses
features_to_process = [col for col in df_sample.columns if col not in time_features]
df_for_processing = df_sample[features_to_process]
# --- AKHIR PERUBAHAN ---

st.info(f"Anda sedang bekerja dengan data dari stasiun: **{st.session_state.data_location}**")
st.warning(f"Untuk tujuan demonstrasi, kami hanya menggunakan **{df_for_processing.shape[1]} fitur non-waktu** dan hanya menampilkan **1000 baris data awal**.", icon="💡")

# --- Tombol untuk Menjalankan Pipeline ---
if st.button("▶️ Jalankan Pipeline Preprocessing", type="primary", use_container_width=True):
    with st.spinner("Menjalankan proses..."):
        station_names = st.session_state.get('locations_name', [st.session_state.data_location])
        station_code_map = {name: i + 1 for i, name in enumerate(station_names)}

        # Gunakan 'df_for_processing' yang sudah difilter
        # Langkah 1: Encoding
        st.session_state.df_encoded = utils.preprocess_encoding(df_for_processing, st.session_state.data_location, station_code_map)
        # Langkah 2: Imputasi
        st.session_state.df_imputed = utils.preprocess_impute(st.session_state.df_encoded)
        # Langkah 3: Normalisasi
        st.session_state.df_scaled, st.session_state.scaler = utils.preprocess_scale(st.session_state.df_imputed)
    st.success("Pipeline Preprocessing Selesai!")

# --- Tampilan Hasil dengan TAB ---
if 'df_scaled' in st.session_state:
    # (Sisa kode untuk menampilkan tab tidak perlu diubah)
    # Cukup pastikan data yang ditampilkan berasal dari df_for_processing atau session state yang sesuai
    tab1, tab2, tab3 = st.tabs(["**Langkah 1: Label Encoding**", "**Langkah 2: Imputasi Nilai Hilang**", "**Langkah 3: Normalisasi Min-Max**"])

    with tab1:
        st.header("Encoding Fitur Kategorikal", divider="blue")
        st.markdown("Mengubah fitur non-numerik seperti arah angin (`wd`) dan nama stasiun (`station`) menjadi nilai numerik.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Encoding")
            st.dataframe(df_for_processing.head(1000)) # Tampilkan 1000 data awal yang sudah difilter
        with col2:
            st.subheader("Sesudah Encoding")
            st.dataframe(utils.highlight_encoded_and_na(st.session_state.df_encoded.head(1000), ['wd', 'station']))

    with tab2:
        st.header("Mengisi Nilai yang Hilang (Imputasi)", divider="blue")
        st.markdown("Menggunakan **interpolasi linear** untuk mengisi celah (nilai `NaN`) dalam data. Perhatikan bagaimana sel yang disorot merah menjadi terisi.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Imputasi")
            st.dataframe(utils.highlight_encoded_and_na(st.session_state.df_encoded.head(1000), []))
        with col2:
            st.subheader("Sesudah Imputasi")
            st.dataframe(st.session_state.df_imputed.head(1000).style.format("{:.2f}"))

    with tab3:
        st.header("Normalisasi Data dengan Min-Max Scaling", divider="blue")
        st.markdown("Menskalakan semua nilai fitur ke dalam rentang **[0, 1]**. Ini memastikan tidak ada satu fitur pun yang mendominasi model hanya karena skalanya lebih besar.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Normalisasi")
            st.dataframe(st.session_state.df_imputed.head(1000).style.format("{:.2f}"))
        with col2:
            st.subheader("Sesudah Normalisasi")
            st.dataframe(utils.highlight_min_max(st.session_state.df_scaled.head(1000), precision=2))