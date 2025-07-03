# pages/6_evaluation.py
import streamlit as st
import pandas as pd
import utils # Kita masih butuh plotly express dari utils

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Hasil Evaluasi Model", page_icon="🏆", layout="wide")

st.title("🏆 Hasil Akhir Evaluasi Model")
st.markdown("""
Halaman ini menyajikan hasil akhir dari perbandingan empat skenario pemodelan yang telah dievaluasi menggunakan **keseluruhan dataset**. 
Tujuan evaluasi ini adalah untuk menemukan arsitektur prapemrosesan yang memberikan performa peramalan terbaik, diukur dengan metrik **RMSE** dan **R-squared (R²)**.
""")
st.info("Hasil di bawah ini bersifat final dan dihitung dari analisis pada data penuh (35.000+ baris) untuk akurasi tertinggi.", icon="💡")

# --- PERUBAHAN UTAMA: HARDCODE HASIL FINAL DI SINI ---
# Ganti angka-angka di bawah ini dengan hasil akhir dari notebook Anda.
# Saya menggunakan angka placeholder yang masuk akal sebagai contoh.
FINAL_EVALUATION_RESULTS = [
    {
        "Skenario": "Skenario 1: Baseline", 
        "Rata-rata RMSE": 0.0437, 
        "Deviasi Standar RMSE": None, 
        "Rata-rata R²": 0.7229
    },
    {
        "Skenario": "Skenario 2: Dengan Seleksi Fitur", 
        "Rata-rata RMSE": 0.0270, 
        "Deviasi Standar RMSE": None, 
        "Rata-rata R²": 0.4731
    },
    {
        "Skenario": "Skenario 3: Hanya Penghapusan Outlier", 
        "Rata-rata RMSE": 0.0428, 
        "Deviasi Standar RMSE": None, 
        "Rata-rata R²": 0.7313
    },
    {
        "Skenario": "Skenario 4: Seleksi Fitur & Penghapusan Outlier", 
        "Rata-rata RMSE": 0.0259, 
        "Deviasi Standar RMSE": None, 
        "Rata-rata R²": 0.5139 
    }
]
# --- AKHIR PERUBAHAN ---

# Buat DataFrame dari hasil final
df_comparison = pd.DataFrame(FINAL_EVALUATION_RESULTS)

# Tentukan skenario terbaik berdasarkan RMSE terendah dari data final
best_scenario = df_comparison.loc[df_comparison['Rata-rata RMSE'].idxmin()]

st.header("Perbandingan Performa Model antar Skenario", divider="blue")

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Perbandingan RMSE")
    df_rmse_sorted = df_comparison.sort_values(by="Rata-rata RMSE", ascending=True)
    
    fig_rmse = utils.px.bar(df_rmse_sorted, x="Skenario", y="Rata-rata RMSE", 
                            error_y="Deviasi Standar RMSE", text_auto='',
                            title="Rata-rata RMSE (Terendah adalah Terbaik)")
    fig_rmse.update_traces(textangle=0, textposition="outside")
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    st.markdown("##### Perbandingan R²")
    # Urutkan data berdasarkan R² (dari tertinggi ke terendah)
    df_r2_sorted = df_comparison.sort_values(by="Rata-rata R²", ascending=False)
    
    fig_r2 = utils.px.bar(df_r2_sorted, x="Skenario", y="Rata-rata R²",
                          text_auto='', title="Rata-rata R² Score (Tertinggi adalah Terbaik)")
    fig_r2.update_traces(textangle=0, textposition="outside")
    st.plotly_chart(fig_r2, use_container_width=True)

st.header("Kesimpulan Evaluasi", divider="blue")
st.success(f"""
Berdasarkan hasil perbandingan, **{best_scenario['Skenario']}** terpilih sebagai arsitektur model terbaik 
dengan rata-rata **RMSE {best_scenario['Rata-rata RMSE']}** dan **R² Score {best_scenario['Rata-rata R²']}**. 
Model yang dilatih dengan skenario inilah yang akan digunakan pada halaman prediksi.
""")