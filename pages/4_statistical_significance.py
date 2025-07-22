# pages/4_statistical_significance.py
import streamlit as st
import pandas as pd
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Signifikansi Statistik", page_icon="🔬", layout="wide")

st.title("🔬 Seleksi _Feature 'Statistical Significance'_")
st.markdown("""
Pada tahap ini, kita melakukan seleksi _feature_ yang paling signifikan dan relevan untuk prediksi. 
Tujuannya adalah untuk meningkatkan performa model dengan mengurangi _feature_ yang tidak relevan dan multikolinearitas (_feature_ yang terlalu mirip satu sama lain).
Ada 2 metode yang digunakan yaitu _Pearson Correlation Coefficient_ (**PCC**) & _Variance Inflation Factor_ (**VIF**)
""")
st.page_link("https://doi.org/10.1016/j.heliyon.2023.e13483", label="Ilu et al. (2021)", icon="🔗")

# --- KEAMANAN & PEMUATAN DATA DARI SESSION STATE ---
if 'df_scaled' not in st.session_state:
    st.warning("🚨 Silakan jalankan pipeline di halaman 'Data Preprocessing' terlebih dahulu untuk melihat halaman ini.")
    st.page_link("pages/3_data_preprocessing.py", label="Kembali ke Halaman Preprocessing", icon="⚙️")
    st.stop()

# Ambil data yang sudah diskalakan dari session state
df_scaled = st.session_state.df_scaled

# --- KONTROL PENGGUNA DI SIDEBAR ---
st.sidebar.header("⚙️ Atur Ambang Batas")
corr_threshold = st.sidebar.slider(
    "Ambang Batas PCC (Pearson Correlation Coefficient):",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Fitur dengan nilai korelasi absolut terhadap PM2.5 di bawah ini akan dihapus."
)
vif_threshold = st.sidebar.slider(
    "Ambang Batas VIF (Variance Inflation Factor):",
    min_value=1.0, max_value=20.0, value=10.0, step=0.5,
    help="Fitur dengan VIF di atas ambang batas ini akan dihapus."
)

# --- Proses Seleksi Fitur Sekuensial ---
corr_matrix = df_scaled.corr(method='pearson')
corr_with_target = corr_matrix['PM2.5'].abs().drop('PM2.5')
features_passed_corr = corr_with_target[corr_with_target >= corr_threshold].index.tolist()
features_failed_corr = corr_with_target[corr_with_target < corr_threshold].index.tolist()

df_for_vif = df_scaled[features_passed_corr]
vif_df = utils.calculate_vif(df_for_vif) if features_passed_corr else pd.DataFrame(columns=['Fitur', 'Skor VIF'])
features_passed_vif = vif_df[vif_df['Skor VIF'] <= vif_threshold]['Fitur'].tolist()
features_failed_vif = vif_df[vif_df['Skor VIF'] > vif_threshold]['Fitur'].tolist()

final_features = features_passed_vif
st.session_state['final_features'] = final_features

# --- Tampilan Utama dengan Tab ---
tab1, tab2, tab3 = st.tabs(["**Tahap 1: Seleksi PCC**", "**Tahap 2: Seleksi VIF**", "**✅ Hasil Akhir**"])

with tab1:
    st.header("Seleksi berdasarkan PCC terhadap PM2.5", divider="blue")
    st.markdown(f"""
    Langkah pertama adalah menemukan _feature_ yang memiliki hubungan paling kuat dengan kolom **PM2.5**. Kita menggunakan **PCC** untuk ini. Feature dengan nilai korelasi di bawah ambang batas **{corr_threshold}** dianggap tidak cukup relevan dan akan dihapus.
    
    - **Latar Hijau:** _feature_ lolos seleksi.
    - **Latar Merah:** _feature_ tidak lolos seleksi.
    """)

    corr_df_display = corr_with_target.reset_index()
    corr_df_display.columns = ['_feature_', 'Korelasi Absolut dengan PM2.5']
    
    def highlight_corr(val, threshold):
        color = '#E8F5E9' if val >= threshold else '#FFCDD2'
        return f'background-color: {color}'

    styled_corr_df = corr_df_display.style.apply(
        lambda x: x.map(lambda v: highlight_corr(v, corr_threshold)), subset=['Korelasi Absolut dengan PM2.5']
    ).format("{:.3f}", subset=['Korelasi Absolut dengan PM2.5'])
    
    st.dataframe(styled_corr_df, use_container_width=True)

    st.subheader(f"Hasil Tahap 1: {len(features_passed_corr)} Fitur Lolos")
    st.success(f"_Feature_ yang akan dianalisis lebih lanjut di tahap VIF: **{', '.join(features_passed_corr) if features_passed_corr else 'Tidak ada'}**")

with tab2:
    st.header("Seleksi berdasarkan VIF (Multikolinearitas)", divider="blue")
    st.markdown(f"""
    Langkah kedua adalah menangani **multikolinearitas**, yaitu kondisi di mana beberapa _feauture_ saling berkorelasi tinggi satu sama lain (tidak termasuk _feature_ yang diprediksi yaitu PM 2.5). Kita menggunakan **Variance Inflation Factor (VIF)** untuk mengukurnya. Feature dengan skor VIF di atas ambang batas **{vif_threshold}** akan dihapus.
    
    - **Latar Hijau:** _feature_ lolos seleksi (VIF rendah).
    - **Latar Merah:** _feature_ tidak lolos seleksi (VIF tinggi).
    """)

    if not features_passed_corr:
        st.warning("Tidak ada _feature_ yang lolos tahap 1 untuk dianalisis VIF.")
    else:
        def highlight_vif(val, threshold):
            color = '#FFCDD2' if val > threshold else '#E8F5E9'
            return f'background-color: {color}'

        styled_vif_df = vif_df.style.apply(lambda x: x.map(lambda v: highlight_vif(v, vif_threshold)), subset=['Skor VIF']).format({"Skor VIF": "{:.2f}"})
        st.dataframe(styled_vif_df, use_container_width=True)

        st.subheader(f"Hasil Tahap 2: {len(features_passed_vif)} Fitur Lolos")
        st.success(f"_Feature_ yang lolos dari seleksi VIF: **{', '.join(features_passed_vif) if features_passed_vif else 'Tidak ada'}**")

with tab3:
    st.header("Ringkasan dan Hasil Akhir Seleksi _Feature_", divider="blue")
    st.markdown("Berikut adalah daftar _feature_ akhir yang akan digunakan untuk prediksi, beserta ringkasan _feature_ yang dihapus pada setiap tahap.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader(f"✅ _Feature_ yang dipertahankan ({len(final_features)})")
            if final_features:
                st.success(", ".join(final_features))
            else:
                st.warning("Tidak ada _feature_ yang lolos seleksi.")
            
    with col2:
        with st.container(border=True):
            st.subheader("❌ _Feature_ yang dihapus")
            if features_failed_corr:
                st.markdown("**Tahap 1 (Korelasi Rendah):**")
                st.error(f"{', '.join(features_failed_corr)}")
            if features_failed_vif:
                st.markdown("**Tahap 2 (VIF Tinggi):**")
                st.error(f"{', '.join(features_failed_vif)}")
            if not features_failed_corr and not features_failed_vif:
                st.info("Tidak ada _feature_ yang dihapus berdasarkan ambang batas saat ini.")

    st.info("_Feature_ yang dipertahankan ini akan disimpan dan digunakan pada tahap pemodelan selanjutnya.", icon="💡")
