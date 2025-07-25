# pages/2_data_overview.py
import streamlit as st
import pandas as pd
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ikhtisar Data", page_icon="📊", layout="wide")

st.title("📊 Ikhtisar dan Tinjauan Data")
st.markdown("""
Selamat datang di halaman ikhtisar data. Halaman ini adalah titik awal untuk memahami karakteristik data _time series_ yang kita gunakan. Silakan pilih stasiun pemantauan untuk memulai eksplorasi.
""")
st.page_link("https://doi.org/10.1098/rspa.2017.0457", label="Zhang et al. (2017)", icon="🔗")

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

# Muat data dan simpan/perbarui session state
if selected_station:
    df = utils.load_station_data(selected_station, num_rows=5000)
    # Logika session state yang disederhanakan
    st.session_state['main_dataframe'] = df
    st.session_state['data_location'] = selected_station
    
    if df is not None:
        # --- TATA LETAK MENGGUNAKAN TAB ---
        tab1, tab2, tab3 = st.tabs(["**Tinjauan Umum**", "**Distribusi Fitur**", "**Hubungan Antar Fitur**"])

        # --- TAB 1: Tinjauan Umum ---
        with tab1:
            st.header("Tampilan Awal Data Mentah", divider="blue")
            st.markdown("Berikut adalah 5 baris pertama dari data untuk memberikan gambaran tentang struktur dan jenis data yang ada.")
            st.dataframe(df.head())
            st.info(f"Data untuk stasiun **{selected_station}** terdiri lebih dari **{df.shape[0]}** baris dan **{df.shape[1]}** kolom.")

            st.header("Deskripsi Setiap Fitur", divider="blue")
            st.markdown("Tabel ini menjelaskan arti dari setiap kolom tabel (**_feature_**) dalam data.")
            descriptions = [(feat, utils.ALL_FEATURE_DESCRIPTIONS.get(feat, "N/A")) for feat in df.columns]
            st.dataframe(pd.DataFrame(descriptions, columns=["Fitur", "Deskripsi"]), use_container_width=True)

            st.header("Analisis Nilai yang Hilang (Missing Values)", divider="blue")
            st.markdown("Grafik batang di bawah ini menunjukkan jumlah data yang hilang (**_missing values_**) untuk setiap _feature_. Feature dengan batang yang tinggi memerlukan penanganan khusus (**_imputation_**) sebelum data di olah.")
            missing_plot = utils.create_missing_values_plot(df)
            if missing_plot:
                st.pyplot(missing_plot)
            else:
                st.success("🎉 Tidak ada nilai yang hilang pada data ini!")

        # --- TAB 2: Distribusi Fitur ---
        with tab2:
            st.header("_Timeseries_ PM2.5", divider="blue")
            st.markdown("Visualisasi ini menunjukkan bagaimana nilai **PM2.5** (data yang akan diprediksi) dari waktu ke waktu. Ini membantu kita melihat adanya kenaikan atau penurunan, musiman, atau anomali (**_outlier_**).")
            timeseries_plot = utils.create_timeseries_plot(df, 'PM2.5')
            st.plotly_chart(timeseries_plot, use_container_width=True)
            
            st.header("Distribusi Masing-Masing Fitur", divider="blue")
            st.markdown("Pilih sebuah fitur untuk melihat distribusi nilainya dalam bentuk histogram. Ini berguna untuk memahami rentang nilai dan apakah distribusinya normal atau miring (skewed).")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            # Pastikan 'PM2.5' ada sebelum mencoba mencari indeksnya
            default_index = numeric_cols.index('PM2.5') if 'PM2.5' in numeric_cols else 0
            feature_to_plot = st.selectbox("Pilih Fitur:", numeric_cols, index=default_index)
            dist_plot = utils.create_distribution_plot(df, feature_to_plot)
            st.plotly_chart(dist_plot, use_container_width=True)

        # --- TAB 3: Hubungan Antar Fitur ---
        with tab3:
            st.header("_Heatmap_ Korelasi Antar Fitur", divider="blue")
            st.markdown("""
            _Heatmap_ ini mengukur kekuatan hubungan linear antara setiap pasang fitur. Arahkan kursor ke sebuah kotak untuk melihat nilai pastinya.
            - **Warna Biru Tua (mendekati +1):** Menunjukkan korelasi positif yang kuat (jika satu naik, yang lain cenderung naik).
            - **Warna Merah Tua (mendekati -1):** Menunjukkan korelasi negatif yang kuat (jika satu naik, yang lain cenderung turun).
            - **Warna Terang (mendekati 0):** Menunjukkan hubungan linear yang lemah.
            """)
            # Asumsikan Anda memiliki fungsi create_correlation_plot_plotly di utils.py
            corr_plot = utils.create_correlation_plot(df) # Menggunakan versi Seaborn/matplotlib
            st.pyplot(corr_plot)

else:
    st.info("Silakan pilih stasiun untuk melanjutkan.")
