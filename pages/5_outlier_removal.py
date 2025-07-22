# pages/5_outlier_removal.py
import streamlit as st
import pandas as pd
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Penghapusan Outlier", page_icon="🗑️", layout="wide")

st.title("🗑️ _Clustering_ untuk Menghapus _Outlier_")
st.markdown("""
_Outlier_ adalah data yang secara signifikan berbeda dari data lainnya. Kehadiran _outlier_ dapat mengganggu kemampuan prediksi, sehingga menghasilkan prediksi yang kurang akurat. 
Tahap ini bertujuan untuk mengidentifikasi dan menghapus _outlier_ menggunakan metode _clustering_ untuk meningkatkan akurasi prediksi.
""")
st.page_link("https://doi.org/10.1016/j.heliyon.2023.e13483", label="Ilu et al. (2021)", icon="🔗")

st.info("""
**Metodologi:** Kita menggunakan algoritma **K-Means dengan perhitungan jarak Mahalanobis**. Ide dasarnya adalah mengelompokkan semua titik data ke dalam beberapa _cluster_. Titik data yang berada sangat jauh dari pusat _cluster_-nya sendiri (memiliki *silhouette score* rendah) dianggap sebagai outlier.
""", icon="💡")


# --- KEAMANAN & PEMUATAN DATA DARI SESSION STATE ---
if 'df_scaled' not in st.session_state or 'final_features' not in st.session_state:
    st.warning("🚨 Silakan jalankan pipeline di halaman 'Data Preprocessing' dan 'Signifikansi Statistik' terlebih dahulu.")
    st.page_link("pages/4_statistical_significance.py", label="Kembali ke Halaman Signifikansi Statistik", icon="🔬")
    st.stop()

# Ambil data yang sudah diskalakan dan fitur final
df_scaled = st.session_state.df_scaled
final_features = st.session_state.final_features
# Analisis outlier dilakukan pada target dan fitur-fitur yang sudah lolos seleksi
df_for_clustering = df_scaled[['PM2.5'] + final_features]

# --- KONTROL PENGGUNA DI SIDEBAR ---
st.sidebar.header("⚙️ Kontrol Analisis Outlier")
k_range = st.sidebar.slider("Rentang 'k' untuk diuji:", min_value=2, max_value=15, value=10, help="Jumlah maksimum klaster yang akan diuji untuk menemukan k optimal.")
k_optimal = st.sidebar.number_input("Pilih 'k' Optimal:", min_value=2, max_value=k_range, value=2, help="Masukkan nilai 'k' yang Anda anggap terbaik berdasarkan grafik di bawah.")
silhouette_threshold = st.sidebar.slider("Ambang Batas _Silhouette Score_:", min_value=-0.5, max_value=0.5, value=0.1, step=0.05, help="Data dengan skor di bawah ini akan dianggap sebagai outlier. Nilai mendekati 0 atau negatif menunjukkan outlier.")

# --- TAHAP 1: MENEMUKAN K OPTIMAL ---
st.header("Tahap 1: Menentukan Jumlah _Cluster_ (k) Optimal", divider="blue")
st.markdown("Langkah pertama adalah menemukan jumlah _cluster_ (`k`) yang paling cocok untuk data kita. Kita menggunakan dua metode visual dari grafik _Elbow_ & grafik _Silhouette_ untuk membantu keputusan ini.")

if st.button("Jalankan Analisis _Cluster_", type="primary"):
    with st.spinner("Menjalankan K-Means untuk setiap k..."):
        sse_scores, silhouette_scores = utils.run_clustering_analysis(df_for_clustering, k_range)
        st.session_state.sse_scores = sse_scores
        st.session_state.silhouette_scores = silhouette_scores

if 'sse_scores' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Method")
        st.markdown("Cari titik di mana penurunan grafik mulai melandai, membentuk 'siku' (_elbow_). Ini adalah kandidat `k` yang baik.")
        elbow_plot = utils.create_elbow_plot(st.session_state.sse_scores)
        st.plotly_chart(elbow_plot, use_container_width=True)
    with col2:
        st.subheader("Silhouette Score Plot")
        st.markdown("Cari nilai `k` yang memberikan skor _silhouette_ tertinggi. Skor yang lebih tinggi menunjukkan _cluster_ yang lebih padat dan terpisah dengan baik.")
        sil_plot = utils.create_silhouette_plot(st.session_state.silhouette_scores)
        st.plotly_chart(sil_plot, use_container_width=True)

# --- TAHAP 2: HASIL CLUSTERING & PENGHAPUSAN OUTLIER ---
st.header(f"Tahap 2: Hasil Clustering dengan k={k_optimal} & Penghapusan Outlier", divider="blue")

if st.session_state.get('sse_scores'): # Hanya tampilkan jika analisis sudah dijalankan
    df_cleaned, df_outliers, df_with_clusters = utils.get_outliers_df(df_for_clustering, k_optimal, silhouette_threshold)
    st.session_state.df_cleaned = df_cleaned # Simpan data bersih untuk halaman berikutnya
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Visualisasi _Cluster_")
        st.markdown(f"Cluster data divisualisasikan menjadi {k_optimal} _cluster_. Setiap warna mewakili satu klaster. Titik data yang ideal seharusnya berkelompok dengan warna yang sama.")
        cluster_plot = utils.create_cluster_visualization(df_with_clusters)
        st.plotly_chart(cluster_plot, use_container_width=True)
    with col2:
        st.subheader("Ringkasan Hasil")
        st.markdown("Berdasarkan parameter yang Anda pilih, berikut adalah ringkasan proses penghapusan _outlier_.")
        st.metric("Total Data Dianalisis", f"{len(df_with_clusters)} baris")
        st.metric("Jumlah _Outlier_ Ditemukan", f"{len(df_outliers)} baris", delta=f"-{len(df_outliers)}", delta_color="inverse")
        st.metric("Data Bersih (_Inliers_)", f"{len(df_cleaned)} baris")

    st.subheader(f"Tabel Data yang Diidentifikasi sebagai _outlier_ (Silhouette Score ≤ {silhouette_threshold})")
    if df_outliers.empty:
        st.success("Tidak ada _outlier_ yang ditemukan dengan ambang batas saat ini.")
    else:
        st.dataframe(df_outliers)
    
    st.success(f"Data yang sudah bersih ({len(df_cleaned)} baris) telah disimpan dan siap untuk tahap evaluasi model.", icon="✅")
