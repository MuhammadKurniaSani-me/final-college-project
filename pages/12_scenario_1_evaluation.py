import streamlit as st
import pandas as pd
import utils

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Evaluasi Skenario 1", page_icon="🧪", layout="wide")

st.title("🧪 Evaluasi Penuh - Skenario 1")
st.header("Model Baseline (Hanya Preprocessing)", divider="blue")
st.markdown("Halaman ini menguji performa Skenario 1 (Baseline). Model dilatih hanya menggunakan prapemrosesan dasar **tanpa seleksi fitur statistik ataupun penghapusan outlier**. Ini berfungsi sebagai titik acuan untuk semua skenario lain.")
st.warning("Proses ini sangat intensif dan mungkin memakan waktu beberapa menit untuk berjalan pertama kali.", icon="⏳")

# Inisialisasi session state jika belum ada
if 'full_eval_results' not in st.session_state:
    st.session_state.full_eval_results = {}

# Tombol untuk memulai evaluasi
SCENARIO_NAME = "Skenario 1"
if st.button(f"Jalankan Evaluasi Penuh untuk {SCENARIO_NAME}", type="primary"):
    with st.status(f"Mengevaluasi '{SCENARIO_NAME}' di 12 stasiun...", expanded=True) as status:
        result_df, logs = utils.run_evaluation_for_all_stations(
            scenario_name=SCENARIO_NAME,
            use_fs=False,   # <-- Parameter Skenario 1
            use_or=False    # <-- Parameter Skenario 1
        )
        st.session_state.full_eval_results[SCENARIO_NAME] = {"results": result_df, "logs": logs}
        # status.update("Evaluasi selesai!", state="complete", expanded=False)
        status.update(label="Evaluasi selesai!", state="complete", expanded=False)

# Tampilkan hasil jika sudah ada
if SCENARIO_NAME in st.session_state.full_eval_results:
    output = st.session_state.full_eval_results[SCENARIO_NAME]
    results_df = output["results"]
    log_data = output["logs"]
    
    st.header("Hasil Evaluasi Lintas Stasiun", divider="blue")
    
    # Tampilkan metrik keseluruhan
    col1, col2 = st.columns(2)
    overall_avg_rmse = results_df['Rata-rata-RMSE'].mean()
    overall_avg_r2 = results_df['Rata-rata-R'].mean()
    col1.metric("Rata-rata RMSE Keseluruhan (12 Stasiun)", f"{overall_avg_rmse:.4f}")
    col2.metric("Rata-rata R-Squared Keseluruhan (12 Stasiun)", f"{overall_avg_r2:.4f}")
    
    # Tampilkan grafik perbandingan
    st.subheader("Performa per Stasiun")
    col_rmse, col_r2 = st.columns(2)
    with col_rmse:
        fig_rmse = utils.px.bar(
            results_df.sort_values(by="Rata-rata-RMSE", ascending=True),
            x="Stasiun", y="Rata-rata-RMSE",
            title="RMSE per Stasiun (Lebih Rendah Lebih Baik)", text_auto='.4f'
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col_r2:
        fig_r2 = utils.px.bar(
            results_df.sort_values(by="Rata-rata-R", ascending=False),
            x="Stasiun", y="Rata-rata-R",
            title="R-Squared per Stasiun (Lebih Tinggi Lebih Baik)", text_auto='.4f'
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Tampilkan log dan tabel detail
    with st.expander("Lihat Log Evaluasi Lengkap"):
        st.code("\n".join(log_data), language="text")
    with st.expander("Lihat Tabel Hasil Detail"):
        st.dataframe(results_df.style.format({"Rata-rata-RMSE": "{:.4f}", "Rata-rata-R": "{:.4f}"}))