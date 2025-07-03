# pages/6_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import utils 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")

st.title("📈 Evaluasi Performa Model")
st.markdown("""
Halaman ini didedikasikan untuk mengevaluasi dan membandingkan performa model ARIMAX di bawah berbagai skenario prapemrosesan data. 
Metode validasi yang digunakan adalah **Repetition Hold-out** dengan 10 repetisi untuk memastikan hasil yang stabil dan andal.
""")

# --- KEAMANAN & PEMUATAN DATA ---
if 'df_cleaned' not in st.session_state or 'df_imputed' not in st.session_state:
    st.warning("🚨 Silakan jalankan pipeline di halaman sebelumnya terlebih dahulu.")
    st.stop()

# st.dataframe(st.session_state.df_scaled)
# st.dataframe(st.session_state.df_cleaned)

# --- DEFINISI SKENARIO ---
scenarios = {
    "Skenario 1: Baseline": {
        "data": st.session_state.df_scaled, # Menggunakan data sebelum outlier removal
        "use_fs": False, "use_or": False
    },
    "Skenario 2: Dengan Seleksi Fitur": {
        "data": st.session_state.df_cleaned,
        "use_fs": True, "use_or": False
    },
    "Skenario 3: Dengan Penghapusan Outlier": {
        "data": st.session_state.df_scaled, # Menggunakan data bersih
        "use_fs": False, "use_or": True # Flag ini sebenarnya untuk logika internal fungsi
    },
    "Skenario 4: Dengan Seleksi Fitur & Penghapusan Outlier": {
        "data": st.session_state.df_cleaned, # Menggunakan data bersih
        "use_fs": True, "use_or": True # Flag ini sebenarnya untuk logika internal fungsi
    },
}

# Inisialisasi session state untuk menyimpan hasil
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = {}

# --- ANTARMUKA PENGGUNA ---
st.sidebar.header("⚙️ Kontrol Evaluasi")
selected_scenario = st.sidebar.selectbox("Pilih Skenario untuk Dievaluasi:", scenarios.keys())

if st.sidebar.button(f"Jalankan Evaluasi untuk {selected_scenario}", type="primary"):
    with st.status(f"Menjalankan evaluasi untuk '{selected_scenario}'...", expanded=True) as status:
        params = scenarios[selected_scenario]
        
        result = utils.run_arimax_evaluation(
            params["data"],
            scenario_name=selected_scenario,
            use_feature_selection=params["use_fs"],
            use_outlier_removal=params["use_or"],
            _session_state=st.session_state,
            _status_container=status
        )
        
        if result:
            st.session_state.eval_results[selected_scenario] = result
            # Perbaikan: Gunakan label="teks pesan"
            status.update(label="Evaluasi Selesai!", state="complete", expanded=False)
        else:
            # Perbaikan: Gunakan label="teks pesan"
            status.update(label="Gagal menjalankan evaluasi.", state="error", expanded=True)

# --- TAMPILAN HASIL ---
st.header("Hasil Evaluasi", divider="blue")

if not st.session_state.eval_results:
    st.info("Pilih skenario dan klik tombol 'Jalankan Evaluasi' di sidebar untuk memulai.")
else:
    # Tampilkan perbandingan semua skenario yang sudah dijalankan
    st.subheader("Perbandingan Rata-rata RMSE antar Skenario")
    
    comparison_data = []
    for name, result in st.session_state.eval_results.items():
        comparison_data.append({
            "Skenario": name,
            "Rata-rata RMSE": result['avg_rmse'],
            "Deviasi Standar": result['std_rmse']
        })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        fig_comp = utils.px.bar(df_comparison, x="Skenario", y="Rata-rata RMSE", 
                                error_y="Deviasi Standar", text_auto='.4f',
                                title="Perbandingan Performa Model")
        fig_comp.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Detail Hasil Skenario Terakhir Dijalankan")
    # Ambil hasil dari skenario terakhir yang dipilih untuk ditampilkan detailnya
    last_run_result = st.session_state.eval_results.get(selected_scenario)
    
    if last_run_result:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rata-rata RMSE (10 Repetisi)", f"{last_run_result['avg_rmse']:.4f}")
        col2.metric("Deviasi Standar RMSE", f"{last_run_result['std_rmse']:.4f}")
        col3.metric("Rata-rata R-Squared (10 Repetisi)", f"{last_run_result['avg_r2']:.4f}")
        col4.metric("Deviasi Standar R-Squared", f"{last_run_result['std_r2']:.4f}")
        
        with st.expander("Lihat Hasil evaluasi per Repetisi"):
            df_scores = pd.DataFrame({
                "Repetisi": range(1, len(last_run_result['all_scores_rmse']) + 1),
                "RMSE": last_run_result['all_scores_rmse'],
                "R-Squared": last_run_result['all_scores_r2']
            })
            st.dataframe(df_scores.style.format({"RMSE": "{:.4f}"}))

        st.subheader("Grafik Prediksi Representatif (dari salah satu repetisi)")
        eval_plot = utils.create_evaluation_plot(last_run_result['plot_data'])
        st.plotly_chart(eval_plot, use_container_width=True)
    else:
        st.info("Hasil untuk skenario yang dipilih akan muncul di sini setelah evaluasi dijalankan.")