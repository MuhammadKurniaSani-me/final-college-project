# pages/6_evaluation.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- FUNGSI UNTUK MEMUAT DATA ---
@st.cache_data
def load_evaluation_data(file_path):
    """Memuat data hasil evaluasi dari file CSV."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File hasil evaluasi tidak ditemukan. Pastikan file '{os.path.basename(file_path)}' ada di dalam folder 'datas/'.")
        return None

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Hasil Evaluasi Model", page_icon="🏆", layout="wide")

st.title("🏆 Hasil Akhir Evaluasi Model")
st.markdown("""
Halaman ini menyajikan hasil akhir dari perbandingan empat skenario pemodelan yang telah dievaluasi menggunakan **keseluruhan dataset (35.000+ baris data)**. 
Tujuan evaluasi ini adalah untuk menemukan arsitektur prapemrosesan yang memberikan performa peramalan terbaik.
""")

# --- BAGIAN 1: PERBANDINGAN KESELURUHAN ---
st.header("Perbandingan Umum antar Skenario", divider="blue")

# Muat data ringkasan utama
SUMMARY_FILE_PATH = "./datas/final-summary.csv"
df_summary = load_evaluation_data(SUMMARY_FILE_PATH)

if df_summary is not None:
    best_scenario = df_summary.loc[df_summary['Rata-rata-RMSE'].idxmin()]
    # best_scenario = df_summary.sort_values(by=["Rata-rata-RMSE", "Rata-rata-R"], ascending=True)

    st.markdown("Grafik di bawah ini membandingkan performa rata-rata dari setiap skenario di 12 stasiun pemantauan.")

    col1, col2 = st.columns(2)
    with col1:
        df_rmse_sorted = df_summary.sort_values(by="Rata-rata-RMSE", ascending=True)
        
        fig_rmse = px.bar(df_rmse_sorted, 
                          x="Skenario", 
                          y="Rata-rata-RMSE", 
                          text_auto='.4f',
                          title="<b>Perbandingan Rata-rata-RMSE</b> (Lebih Rendah Lebih Baik)",
                          labels={"Rata-rata RMSE": "Rata-rata RMSE (Skala Ternormalisasi)"})
        fig_rmse.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        df_r2_sorted = df_summary.sort_values(by="Rata-rata-R", ascending=True)
        
        fig_r2 = px.bar(df_r2_sorted, 
                        x="Skenario", 
                        y="Rata-rata-R", 
                        text_auto='.4f', 
                        title="<b>Perbandingan Rata-rata-R</b> (Lebih Tinggi Lebih Baik)",
                        labels={"Rata-rata-R": "Rata-rata R² Score"})
        fig_r2.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig_r2, use_container_width=True)

    # st.success(f"Secara keseluruhan, **{best_scenario['Skenario']}** terpilih sebagai arsitektur model terbaik.")

# --- BAGIAN 2: DETAIL PER SKENARIO ---
st.header("Detail Performa per Skenario", divider="blue")
st.markdown("Pilih sebuah skenario untuk melihat rincian performa model di setiap stasiun pemantauan.")

# Definisikan path file untuk setiap skenario
# Pastikan Anda mengganti nama file CSV Anda agar sesuai dengan ini
scenario_files = {
    "Skenario 1: Baseline": "./datas/summary_1.csv",
    "Skenario 2: Dengan Seleksi Fitur": "./datas/summary_2.csv",
    "Skenario 3: Hanya Penghapusan Outlier": "./datas/summary_3.csv",
    "Skenario 4: Seleksi Fitur & Penghapusan Outlier": "./datas/summary_4.csv"
}

selected_scenario_for_detail = st.selectbox("Pilih Skenario untuk Dilihat Rinciannya:", list(scenario_files.keys()))

if selected_scenario_for_detail:
    detail_file_path = scenario_files[selected_scenario_for_detail]
    df_detail = load_evaluation_data(detail_file_path)

    if df_detail is not None:
        st.subheader(f"Rincian Performa untuk {selected_scenario_for_detail}")
        
        col_detail_1, col_detail_2 = st.columns(2)
        with col_detail_1:
            df_detail_rmse_sorted = df_detail.sort_values(by="Rata-rata-RMSE", ascending=True)
            fig_detail_rmse = px.bar(df_detail_rmse_sorted, x="Stasiun", y="Rata-rata-RMSE",
                                     title="RMSE per Stasiun", text_auto='.4f')
            st.plotly_chart(fig_detail_rmse, use_container_width=True)
        
        with col_detail_2:
            df_detail_r2_sorted = df_detail.sort_values(by="Rata-rata-R", ascending=False)
            fig_detail_r2 = px.bar(df_detail_r2_sorted, x="Stasiun", y="Rata-rata-R",
                                   title="R² per Stasiun", text_auto='.4f')
            st.plotly_chart(fig_detail_r2, use_container_width=True)
            
        with st.expander("Lihat Tabel Data Rincian"):
            st.dataframe(df_detail.style.format({
                "Rata-rata-RMSE": "{:.4f}",
                "Rata-rata-R": "{:.4f}"
            }))
