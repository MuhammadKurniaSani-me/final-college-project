# pages/1_introduction.py
import streamlit as st

# Atur konfigurasi halaman untuk tampilan yang konsisten
st.set_page_config(page_title="Introduction", page_icon="👋", layout="wide")

def introduction_section():
    """
    Menampilkan bagian pengantar utama dengan konteks historis dan tautan referensi.
    """
    st.header("Sejarah Analisis Runtun Waktu", divider="grey")
    st.markdown("""
    Sejak awal tahun 1990-an, analisis data runtun waktu, terutama yang berkaitan dengan kualitas udara, telah menjadi bidang penelitian yang signifikan. 
    Model-model awal telah membuka jalan bagi teknik-teknik yang lebih canggih yang digunakan saat ini. Di bawah ini adalah beberapa studi modern relevan yang dibangun di atas sejarah ini.
    """)
    
    # Menggunakan kolom untuk tata letak tautan yang lebih rapi
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("https://doi.org/10.3390/su151813951", label="Abimannan et al. (2023)", icon="🔗")
    with col2:
        st.page_link("https://doi.org/10.1007/s10462-023-10424-4", label="Méndez et al. (2023)", icon="💡")
    
    # Menggunakan st.divider() untuk pemisah visual yang bersih
    st.divider()

def introduction_key_points():
    """
    Menampilkan poin-poin kunci tentang analisis kualitas udara dengan tombol tautan eksternal.
    """
    st.header("Poin-Poin Kunci", divider="grey")

    col1, col2, col3, col4 = st.columns(4)

    # --- Kartu 1 ---
    with col1:
        with st.container(border=True, height=300):
            st.markdown("##### 🏙️ Lingkungan Kota Besar")
            st.write(
                "Kota-kota besar seperti Beijing, Tiongkok, telah menjadi titik fokus untuk mempelajari polusi udara yang tinggi, mendorong penelitian di bidang ini."
            )
            # PERUBAHAN: Menggunakan st.link_button untuk tautan eksternal
            st.link_button("Peringkat Kualitas Udara Dunia", "https://www.iqair.com/world-air-quality-ranking", use_container_width=True)

    # --- Kartu 2 ---
    with col2:
        with st.container(border=True, height=300):
            st.markdown("##### ⏳ Data Runtun Waktu yang Kompleks")
            st.write(
                "Data runtun waktu untuk kualitas udara terus bertambah volumenya dan dipengaruhi oleh banyak faktor eksternal yang kompleks."
            )
            # PERUBAHAN: Menggunakan st.link_button untuk tautan eksternal
            st.link_button("Lihat Dataset", "https://doi.org/10.24432/C5RK5G", use_container_width=True)

    # --- Kartu 3 ---
    with col3:
        with st.container(border=True, height=300):
            st.markdown("##### ⚙️ Model ARIMA yang Kuat")
            st.write(
                "Model ARIMA sederhana namun kuat. Model ini dapat diperluas menjadi ARIMAX atau SARIMA untuk menyertakan faktor musiman dan eksternal."
            )
            # PERUBAHAN: Menggunakan st.link_button untuk tautan eksternal
            st.link_button("Pengantar Model ARIMA", "https://medium.com/analytics-vidhya/a-thorough-introduction-to-arima-models-987a24e9ff71", use_container_width=True)

    # --- Kartu 4 ---
    with col4:
        with st.container(border=True, height=300):
            st.markdown("##### 🧩 Metode Hybrid & Ensemble")
            st.write(
                "Pendekatan modern sering menggabungkan model (metode hybrid) atau menggunakan pra-pemrosesan canggih untuk meningkatkan akurasi peramalan."
            )
            # PERUBAHAN: Menggunakan st.link_button untuk tautan eksternal
            st.link_button("Baca Paper Terkait", "https://doi.org/10.1016/j.heliyon.2023.e13483", use_container_width=True)


# --- Eksekusi Utama Halaman ---
st.title("👋 Pengantar Peramalan Kualitas Udara")
st.write("Halaman ini memberikan gambaran umum mengenai konteks dan konsep kunci dalam proyek analisis dan peramalan kualitas udara.")
st.divider()

# Menampilkan konten
introduction_section()
introduction_key_points()