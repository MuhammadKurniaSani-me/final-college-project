import streamlit as st
import utils


# Atur konfigurasi halaman untuk tampilan yang konsisten
st.set_page_config(page_title="Introduction", page_icon="👋")

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
        st.page_link("https://doi.org/10.1007/s10462-023-10424-4", label="Méndez et al. (2023)", icon="🔗")
    
    # Menggunakan st.divider() untuk pemisah visual yang bersih
    st.divider()

def introduction_key_points():
    """
    Menampilkan poin-poin kunci tentang analisis kualitas udara dalam tata letak kartu 4 kolom yang modern.
    """
    st.header("Poin-Poin Kunci", divider="grey")

    col1, col2, col3, col4 = st.columns(4)

    # --- Kartu 1 ---
    with col1:
        # Menggunakan st.container dengan border=True untuk efek kartu
        with st.container(border=True, height=300):
            st.markdown("##### 🏙️ Lingkungan Kota Besar")
            st.write(
                "Kota-kota besar seperti Beijing, Tiongkok, telah menjadi titik fokus untuk mempelajari polusi udara yang tinggi, mendorong penelitian di bidang ini."
            )
            if st.button("5 Kota Paling Berpolusi", key="b1"):
                # Di aplikasi nyata, ini dapat menavigasi atau menampilkan data
                # https://www.iqair.com/world-air-quality-ranking
                st.info("Menampilkan 5 kota paling berpolusi...") 

    # --- Kartu 2 ---
    with col2:
        with st.container(border=True, height=300):
            st.markdown("##### ⏳ Data Runtun Waktu yang Kompleks")
            st.write(
                "Data runtun waktu untuk kualitas udara terus bertambah volumenya dan dipengaruhi oleh banyak faktor eksternal yang kompleks."
            )
            if st.button("Lihat Kompleksitas Data", key="b2"):
                # https://doi.org/10.24432/C5RK5G
                st.info("Menampilkan faktor-faktor kompleksitas data...")

    # --- Kartu 3 ---
    with col3:
        with st.container(border=True, height=300):
            st.markdown("##### ⚙️ Model ARIMA yang Kuat")
            st.write(
                "Model ARIMA sederhana namun kuat. Model ini dapat diperluas menjadi ARIMAX atau SARIMA untuk menyertakan faktor musiman dan eksternal."
            )
            if st.button("Jelajahi Model", key="b3"):
                # https://medium.com/analytics-vidhya/a-thorough-introduction-to-arima-models-987a24e9ff71
                st.info("Menampilkan berbagai model runtun waktu...")

    # --- Kartu 4 ---
    with col4:
        with st.container(border=True, height=300):
            st.markdown("##### 🧩 Metode Hybrid & Ensemble")
            st.write(
                "Pendekatan modern sering menggabungkan model (metode hybrid) atau menggunakan pra-pemrosesan canggih untuk meningkatkan akurasi peramalan."
            )
            if st.button("Pelajari Metode Hybrid", key="b4"):
                # https://doi.org/10.1016/j.heliyon.2023.e13483
                st.info("Menampilkan contoh metode hybrid...")


# --- Eksekusi Utama Halaman ---
st.title("👋 Pengantar Peramalan Kualitas Udara")
st.write("Halaman ini memberikan gambaran umum mengenai konteks dan konsep kunci dalam proyek analisis dan peramalan kualitas udara.")
st.divider()

# Menampilkan konten
introduction_section()
introduction_key_points()


# def introduction_section():
#     """
#     """

#     st.header("History of Time Series Analysis", divider="grey")
#     st.markdown("Since 1993, analysis of time series data, especially air quality data, has been carried out.")
#     st.page_link("https://doi.org/10.3390/su151813951", label="Abimannan et.al 2023", icon=":material/open_in_new:")
#     st.page_link("https://doi.org/10.1007/s10462-023-10424-4", label="Méndez et al. 2023", icon=":material/open_in_new:")
#     st.markdown("   ") 


# def introduction_key_points():
#     """
#     """

#     # --- KEY TAKEAWAYS (4 COLUMNS) ---
#     col1, col2, col3, col4 = st.columns(4)

#     # --- COLUMN 1: OpenAI Dominance ---
#     with col1:
#         st.markdown("### Big City's environment is dominance")
#         st.write(
#             "Big city like Beijing in China has become on of the many city with high pollution air"
#         )
#         if st.button("Top 5 big cities", key="b1"):
#             st.info("Redirecting to top models...")

#     # --- COLUMN 2: Multi-Agent Future ---
#     with col2:
#         st.markdown("### Time series data is complex and diverse")
#         st.write(
#             "Time series data increases its volume and time point with some external factor that influence it"
#         )
#         if st.button("See more data", key="b2"):
#             st.info("Redirecting to orchestration tools...")

#     # --- COLUMN 3: Vector Retrieval ---
#     with col3:
#         st.markdown("### ARIMA is simple, powerful, and fast")
#         st.write(
#             "AR and MA model can combined to ARMA, ARIMA, SARIMA, or with Exogenous Factors"
#         )
#         if st.button("See more model", key="b3"):
#             st.info("Redirecting to retrieval tools...")

#     # --- COLUMN 4: Rise of Chatbots ---
#     with col4:
#         st.markdown("### Hybrid method & ensemble learning")
#         st.write(
#             "Additional method like pre-processing or any machine learning algorithm"
#             # Survey:Time-series data preprocessing: A survey and an empirical analysis
#         )
#         if st.button("See more methods", key="b4"):
#             st.info("Exploring the future of chatbots...")

#     st.markdown("   ") 


# # display content
# introduction_section()
# introduction_key_points()
