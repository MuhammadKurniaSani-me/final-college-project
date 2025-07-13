# pages/4_statistical_significance.py
import streamlit as st
import pandas as pd
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Signifikansi Statistik", page_icon="🔬", layout="wide")

st.title("🔬 Alur Seleksi Fitur Sekuensial")
st.markdown("""
Pada tahap ini, kita melakukan seleksi fitur untuk memilih variabel yang paling signifikan dan relevan untuk model peramalan. 
Tujuannya adalah untuk meningkatkan performa model dengan mengurangi *noise* (fitur yang tidak relevan) dan multikolinearitas (fitur yang terlalu mirip satu sama lain).
""")

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
    "Ambang Batas Korelasi Pearson (Absolut):",
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
tab1, tab2, tab3 = st.tabs(["**Tahap 1: Seleksi Korelasi**", "**Tahap 2: Seleksi VIF**", "**✅ Hasil Akhir**"])

with tab1:
    st.header("Seleksi Berdasarkan Korelasi dengan PM2.5", divider="blue")
    st.markdown(f"""
    Langkah pertama adalah menemukan fitur yang memiliki hubungan linear paling kuat dengan target kita, yaitu **PM2.5**. Kita menggunakan **Korelasi Pearson** untuk ini. Fitur dengan korelasi absolut di bawah ambang batas **{corr_threshold}** dianggap tidak cukup relevan dan akan dihapus.
    
    - **Latar Hijau:** Fitur lolos seleksi.
    - **Latar Merah:** Fitur tidak lolos seleksi.
    """)

    corr_df_display = corr_with_target.reset_index()
    corr_df_display.columns = ['Fitur', 'Korelasi Absolut dengan PM2.5']
    
    def highlight_corr(val, threshold):
        color = '#E8F5E9' if val >= threshold else '#FFCDD2'
        return f'background-color: {color}'

    styled_corr_df = corr_df_display.style.apply(
        lambda x: x.map(lambda v: highlight_corr(v, corr_threshold)), subset=['Korelasi Absolut dengan PM2.5']
    ).format("{:.3f}", subset=['Korelasi Absolut dengan PM2.5'])
    
    st.dataframe(styled_corr_df, use_container_width=True)

    st.subheader(f"Hasil Tahap 1: {len(features_passed_corr)} Fitur Lolos")
    st.success(f"Fitur yang akan dianalisis lebih lanjut di tahap VIF: **{', '.join(features_passed_corr) if features_passed_corr else 'Tidak ada'}**")

with tab2:
    st.header("Seleksi Berdasarkan VIF (Multikolinearitas)", divider="blue")
    st.markdown(f"""
    Langkah kedua adalah menangani **multikolinearitas**, yaitu kondisi di mana beberapa fitur independen saling berkorelasi tinggi satu sama lain. Ini dapat mengganggu model. Kita menggunakan **Variance Inflation Factor (VIF)** untuk mengukurnya. Fitur dengan skor VIF di atas ambang batas **{vif_threshold}** akan dihapus.
    
    - **Latar Hijau:** Fitur lolos seleksi (VIF rendah).
    - **Latar Merah:** Fitur tidak lolos seleksi (VIF tinggi).
    """)

    if not features_passed_corr:
        st.warning("Tidak ada fitur yang lolos tahap 1 untuk dianalisis VIF.")
    else:
        def highlight_vif(val, threshold):
            color = '#FFCDD2' if val > threshold else '#E8F5E9'
            return f'background-color: {color}'

        styled_vif_df = vif_df.style.apply(lambda x: x.map(lambda v: highlight_vif(v, vif_threshold)), subset=['Skor VIF']).format({"Skor VIF": "{:.2f}"})
        st.dataframe(styled_vif_df, use_container_width=True)

        st.subheader(f"Hasil Tahap 2: {len(features_passed_vif)} Fitur Lolos")
        st.success(f"Fitur yang lolos dari seleksi VIF: **{', '.join(features_passed_vif) if features_passed_vif else 'Tidak ada'}**")

with tab3:
    st.header("Ringkasan dan Hasil Akhir Seleksi Fitur", divider="blue")
    st.markdown("Berikut adalah daftar fitur final yang akan digunakan untuk pemodelan, beserta ringkasan fitur yang dihapus pada setiap tahap.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader(f"✅ Fitur Final Dipertahankan ({len(final_features)})")
            if final_features:
                st.success(", ".join(final_features))
            else:
                st.warning("Tidak ada fitur yang lolos seleksi.")
            
    with col2:
        with st.container(border=True):
            st.subheader("❌ Fitur yang Dihapus")
            if features_failed_corr:
                st.markdown("**Tahap 1 (Korelasi Rendah):**")
                st.error(f"{', '.join(features_failed_corr)}")
            if features_failed_vif:
                st.markdown("**Tahap 2 (VIF Tinggi):**")
                st.error(f"{', '.join(features_failed_vif)}")
            if not features_failed_corr and not features_failed_vif:
                st.info("Tidak ada fitur yang dihapus berdasarkan ambang batas saat ini.")

    st.info("Fitur final yang dipertahankan ini akan disimpan dan digunakan pada tahap pemodelan selanjutnya.", icon="💡")
