# pages/8_conclusion.py
import streamlit as st

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Kesimpulan", page_icon="🏁", layout="wide")

# --- JUDUL HALAMAN ---
st.title("🏁 Kesimpulan & Ringkasan Proyek")
st.markdown("Halaman ini merangkum temuan akhir dari proyek peramalan PM2.5, menyajikan hasil dari model dengan performa terbaik, serta membahas implikasi dan potensi pengembangannya di masa depan.")
st.divider()

# --- TEMUAN UTAMA (BAGIAN HARDCODED) ---
st.header("🏆 Temuan Utama & Performa Model Terbaik")

# --- UPDATE THESE VALUES WITH YOUR FINAL RESULTS ---

# Nama skenario terbaik Anda.
best_scenario_name = "Skenario 4: Seleksi Fitur & Penghapusan Outlier"

# Metrik performa final dari skenario terbaik.
best_rmse = 0.0339
best_r2 = 0.7508

# Daftar fitur final yang digunakan oleh model terbaik Anda.
final_features_list = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']

# --- AKHIR BAGIAN HARDCODED ---


# Tampilkan kesimpulan statis
st.success(f"Berdasarkan evaluasi yang komprehensif, **{best_scenario_name}** teridentifikasi sebagai pendekatan pemodelan yang paling optimal.")

# Tampilkan metrik utama dari skenario terbaik
col1, col2, col3 = st.columns(3)
col1.metric("Skenario Terbaik", best_scenario_name)
col2.metric("Rata-rata RMSE Final", f"{best_rmse:.4f}", help="Lebih rendah lebih baik.")
col3.metric("Rata-rata R² Score Final", f"{best_r2:.4f}", help="Lebih tinggi (mendekati 1) lebih baik.")

# Tampilkan fitur final yang digunakan
st.info(f"**Fitur Final yang Digunakan Model:** `{', '.join(final_features_list)}`", icon="✅")

st.divider()

# --- IMPLIKASI & KETERBATASAN ---
col_implikasi, col_keterbatasan = st.columns(2)

with col_implikasi:
    with st.container(border=True):
        st.subheader("💡 Implikasi & Manfaat")
        st.markdown("""
        - **Sistem Peringatan Dini:** Model yang dikembangkan dapat menjadi dasar bagi sistem peringatan dini kualitas udara untuk masyarakat di area terkait.
        - **Wawasan Kebijakan:** Analisis fitur memberikan wawasan bagi pembuat kebijakan mengenai polutan mana yang paling berpengaruh dan perlu diprioritaskan dalam penanganan.
        - **Kontribusi Akademis:** Metodologi prapemrosesan yang komprehensif secara kuantitatif menunjukkan bahwa seleksi fitur dan penghapusan outlier dapat meningkatkan akurasi model secara signifikan.
        """)

with col_keterbatasan:
    with st.container(border=True):
        st.subheader("⚠️ Keterbatasan Model")
        st.markdown("""
        - **Data Satu Kota:** Model dilatih secara eksklusif pada data dari Beijing, sehingga generalisasinya ke kota lain mungkin terbatas.
        - **Kejadian Tak Terduga:** Model ARIMAX mungkin kesulitan memprediksi lonjakan polusi ekstrem yang disebabkan oleh kejadian tak terduga (misalnya, kebakaran hutan besar).
        - **Asumsi Linearitas:** Model ARIMA pada dasarnya mengasumsikan adanya hubungan linear, sementara dinamika polusi di dunia nyata bisa jadi sangat non-linear.
        """)

# --- ARAH PENGEMBANGAN SELANJUTNYA ---
st.header("🚀 Arah Pengembangan Selanjutnya")
with st.expander("Klik untuk melihat ide-ide pengembangan di masa depan"):
    st.markdown("""
    1.  **Model Tingkat Lanjut:** Mengimplementasikan model *deep learning* seperti **LSTM (Long Short-Term Memory)** atau **Transformer** untuk menangkap pola non-linear jangka panjang dengan lebih baik.
    2.  **Sumber Data Tambahan:** Mengintegrasikan data tambahan seperti citra satelit (Aerosol Optical Depth), data lalu lintas, atau data aktivitas industri untuk memperkaya fitur eksogen.
    3.  **Implementasi Real-time:** Mengembangkan sistem menjadi aplikasi yang berjalan secara *real-time*, mengambil data terbaru secara otomatis dan memberikan peramalan setiap jam.
    4.  **Analisis Multi-Kota:** Memperluas analisis untuk membandingkan dan memodelkan kualitas udara di beberapa kota secara bersamaan untuk menemukan pola yang lebih umum.
    """)
