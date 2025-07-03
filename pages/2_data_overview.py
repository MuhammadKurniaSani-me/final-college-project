# pages/2_data_overview.py
import streamlit as st
import pandas as pd
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ikhtisar Data", page_icon="📊", layout="wide")

st.title("📊 Ikhtisar dan Tinjauan Data")
st.markdown("""
Halaman ini memungkinkan Anda untuk menjelajahi dataset kualitas udara Beijing secara mendalam. 
Pilih stasiun pemantauan untuk memulai analisis.
""")

# --- PEMILIHAN STASIUN & PEMUATAN DATA ---
STATION_NAMES = [
    'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 
    'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'
]
selected_station = st.selectbox(
    "Pilih Stasiun Pemantauan:",
    STATION_NAMES,
    help="Data akan dimuat ulang sesuai dengan stasiun yang Anda pilih."
)

if selected_station:
    df = utils.load_station_data(selected_station)

    if 'main_dataframe' not in st.session_state:
        st.session_state['main_dataframe'] = df
        st.session_state['data_location'] = selected_station
    
    if df is not None:
        # --- TATA LETAK MENGGUNAKAN TAB ---
        tab1, tab2, tab3 = st.tabs(["**Ikhtisar Data**", "**Distribusi Fitur**", "**Hubungan Antar Fitur**"])

        # --- TAB 1: IKHTISAR DATA ---
        with tab1:
            st.header("Tampilan Awal Data", divider="blue")
            st.dataframe(df.head())
            st.info(f"Dataset untuk stasiun **{selected_station}** terdiri dari **{df.shape[0]}** baris dan **{df.shape[1]}** kolom.")

            st.header("Deskripsi Fitur", divider="blue")
            descriptions = [(feat, utils.ALL_FEATURE_DESCRIPTIONS.get(feat, "N/A")) for feat in df.columns]
            st.dataframe(pd.DataFrame(descriptions, columns=["Fitur", "Deskripsi"]), use_container_width=True)

            st.header("Statistik Nilai yang Hilang (Missing Values)", divider="blue")
            missing_plot = utils.create_missing_values_plot(df)
            if missing_plot:
                st.pyplot(missing_plot)
            else:
                st.success("🎉 Tidak ada nilai yang hilang (missing values) pada dataset ini!")

        # --- TAB 2: DISTRIBUSI FITUR ---
        with tab2:
            st.header("Tren Runtun Waktu PM2.5", divider="blue")
            st.markdown("Visualisasi ini menunjukkan bagaimana nilai PM2.5 berubah dari waktu ke waktu.")
            timeseries_plot = utils.create_timeseries_plot(df, 'PM2.5')
            st.plotly_chart(timeseries_plot, use_container_width=True)
            
            st.header("Distribusi Masing-Masing Fitur", divider="blue")
            st.markdown("Pilih fitur untuk melihat distribusi datanya dalam bentuk histogram.")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            feature_to_plot = st.selectbox("Pilih Fitur:", numeric_cols, index=numeric_cols.index('PM2.5'))
            dist_plot = utils.create_distribution_plot(df, feature_to_plot)
            st.plotly_chart(dist_plot, use_container_width=True)

        # --- TAB 3: HUBUNGAN ANTAR FITUR ---
        with tab3:
            st.header("Heatmap Korelasi", divider="blue")
            st.markdown("""
            Heatmap ini mengukur hubungan linear antara setiap pasang fitur. 
            - Nilai mendekati **+1** (biru tua) berarti korelasi positif yang kuat.
            - Nilai mendekati **-1** (merah tua) berarti korelasi negatif yang kuat.
            - Nilai mendekati **0** berarti hubungan linear yang lemah.
            """)
            corr_plot = utils.create_correlation_plot(df)
            st.pyplot(corr_plot)

else:
    st.info("Silakan pilih stasiun untuk melanjutkan.")