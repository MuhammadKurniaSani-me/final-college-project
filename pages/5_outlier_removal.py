# pages/5_outlier_removal.py
import streamlit as st
import pandas as pd
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Penghapusan Outlier", page_icon="🗑️", layout="wide")

st.title("🗑️ Deteksi dan Penghapusan Outlier")

# --- PERUBAHAN TEKS DI SINI ---
st.markdown("""
Outlier adalah titik data yang secara signifikan berbeda dari observasi lainnya. 
Tahap ini bertujuan untuk mengidentifikasi dan menghapus outlier menggunakan metode clustering **K-Means dengan Mahalanobis Distance**. 
Metrik ini memperhitungkan korelasi antar fitur, sehingga lebih baik dalam menangani data yang tidak berbentuk lingkaran (non-spherical).
""")
# --- AKHIR PERUBAHAN ---

# ... SISA KODE DI HALAMAN INI TIDAK PERLU DIUBAH SAMA SEKALI ...
# ... Karena semua panggilan ke utils.run_clustering_analysis dan utils.get_outliers_df ...
# ... sekarang secara otomatis akan menggunakan implementasi Mahalanobis yang baru.

# --- KEAMANAN & PEMUATAN DATA DARI SESSION STATE ---
if 'df_scaled' not in st.session_state or 'final_features' not in st.session_state:
    st.warning("🚨 Silakan jalankan pipeline di halaman 'Data Preprocessing' dan 'Signifikansi Statistik' terlebih dahulu.")
    st.stop()

df_scaled = st.session_state.df_scaled
final_features = st.session_state.final_features
df_for_clustering = df_scaled[['PM2.5'] + final_features]

st.sidebar.header("⚙️ Kontrol Analisis Outlier")
k_range = st.sidebar.slider("Rentang 'k' untuk diuji:", min_value=2, max_value=15, value=10, help="Jumlah maksimum klaster yang akan diuji untuk menemukan k optimal.")
k_optimal = st.sidebar.number_input("Pilih 'k' Optimal:", min_value=2, max_value=k_range, value=2, help="Masukkan nilai 'k' yang Anda anggap terbaik berdasarkan grafik di bawah.")
silhouette_threshold = st.sidebar.slider("Ambang Batas Silhouette Score:", min_value=-1.0, max_value=1.0, value=0.5, step=0.05, help="Data dengan skor di bawah ini akan dianggap sebagai outlier.")

st.header("Tahap 1: Menentukan Jumlah Klaster (k) Optimal", divider="blue")
if st.button("Jalankan Analisis Klaster (Mahalanobis)", type="primary"):
    with st.spinner("Menjalankan K-Means Mahalanobis untuk setiap k..."):
        sse_scores, silhouette_scores = utils.run_clustering_analysis(df_for_clustering, k_range)
        st.session_state.sse_scores = sse_scores
        st.session_state.silhouette_scores = silhouette_scores

if 'sse_scores' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Method (SSE Mahalanobis)")
        elbow_plot = utils.create_elbow_plot(st.session_state.sse_scores)
        st.plotly_chart(elbow_plot, use_container_width=True)
    with col2:
        st.subheader("Silhouette Score Plot")
        sil_plot = utils.create_silhouette_plot(st.session_state.silhouette_scores)
        st.plotly_chart(sil_plot, use_container_width=True)
    st.info("**Panduan:** Cari 'siku' pada grafik Elbow Method, atau puncak tertinggi pada grafik Silhouette Score untuk menemukan 'k' yang paling optimal.", icon="💡")

st.header(f"Tahap 2: Hasil Clustering dengan k={k_optimal} & Penghapusan Outlier", divider="blue")
if st.session_state.get('sse_scores'): # Hanya tampilkan jika analisis sudah dijalankan
    df_cleaned, df_outliers, df_with_clusters = utils.get_outliers_df(df_for_clustering, k_optimal, silhouette_threshold)
    st.session_state.df_cleaned = df_cleaned
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Visualisasi Klaster")
        st.markdown("Data direduksi menjadi 2 dimensi menggunakan PCA untuk visualisasi.")
        cluster_plot = utils.create_cluster_visualization(df_with_clusters)
        st.plotly_chart(cluster_plot, use_container_width=True)
    with col2:
        st.subheader("Ringkasan Outlier")
        st.metric("Total Data Dianalisis", f"{len(df_with_clusters)} baris")
        st.metric("Jumlah Outlier Ditemukan", f"{len(df_outliers)} baris", delta=f"-{len(df_outliers)}", delta_color="inverse")
        st.metric("Data Bersih (Inliers)", f"{len(df_cleaned)} baris", delta=f"{len(df_cleaned)}", delta_color="normal")

    st.subheader(f"Tabel Data yang Diidentifikasi sebagai Outlier (Silhouette Score < {silhouette_threshold})")
    if df_outliers.empty:
        st.success("Tidak ada outlier yang ditemukan dengan ambang batas saat ini.")
    else:
        st.dataframe(df_outliers)