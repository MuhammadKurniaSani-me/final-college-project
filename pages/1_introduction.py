import streamlit as st

import streamlit as st


def introduction_section():
    """
    """

    st.header("History of Time Series Analysis", divider="grey")
    st.markdown("Since 1993, analysis of time series data, especially air quality data, has been carried out.")
    st.page_link("https://doi.org/10.3390/su151813951", label="Abimannan et.al 2023", icon=":material/open_in_new:")
    st.page_link("https://doi.org/10.1007/s10462-023-10424-4", label="Méndez et al. 2023", icon=":material/open_in_new:")
    st.markdown("   ") 


def introduction_key_points():
    """
    """

    # --- KEY TAKEAWAYS (4 COLUMNS) ---
    col1, col2, col3, col4 = st.columns(4)

    # --- COLUMN 1: OpenAI Dominance ---
    with col1:
        st.markdown("### Big City's environment is dominance")
        st.write(
            "Big city like Beijing in China has become on of the many city with high pollution air"
        )
        if st.button("Top 5 big cities", key="b1"):
            st.info("Redirecting to top models...")

    # --- COLUMN 2: Multi-Agent Future ---
    with col2:
        st.markdown("### Time series data is complex and diverse")
        st.write(
            "Time series data increases its volume and time point with some external factor that influence it"
        )
        if st.button("See more data", key="b2"):
            st.info("Redirecting to orchestration tools...")

    # --- COLUMN 3: Vector Retrieval ---
    with col3:
        st.markdown("### ARIMA is simple, powerful, and fast")
        st.write(
            "AR and MA model can combined to ARMA, ARIMA, SARIMA, or with Exogenous Factors"
        )
        if st.button("See more model", key="b3"):
            st.info("Redirecting to retrieval tools...")

    # --- COLUMN 4: Rise of Chatbots ---
    with col4:
        st.markdown("### Hybrid method & ensemble learning")
        st.write(
            "Additional method like pre-processing or any machine learning algorithm"
            # Survey:Time-series data preprocessing: A survey and an empirical analysis
        )
        if st.button("See more methods", key="b4"):
            st.info("Exploring the future of chatbots...")

    st.markdown("   ") 


# display content
introduction_section()
introduction_key_points()
