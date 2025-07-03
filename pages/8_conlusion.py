# pages/8_conclusion.py
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Conclusion", page_icon="🏁", layout="wide")

# --- PAGE TITLE ---
st.title("🏁 Conclusion & Project Summary")
st.markdown("This page summarizes the final findings of the PM2.5 forecasting project, presenting the results from the best-performing model and discussing its implications and future potential.")
st.divider()

# --- KEY FINDINGS (HARDCODED SECTION) ---
st.header("🏆 Key Findings & Best Model Performance")

# --- UPDATE THESE VALUES WITH YOUR FINAL RESULTS ---

# The name of your best-performing scenario.
best_scenario_name = "Skenario 4: Seleksi Fitur & Penghapusan Outlier"

# The final performance metrics for the best scenario.
best_rmse = 0.0259
best_r2 = 0.5139

# The final list of features used by the best model.
# (Update this list based on your final feature selection results)
final_features_list = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']

# --- END OF HARDCODED SECTION ---


# Display the static conclusion
st.success(f"Based on a comprehensive evaluation, **{best_scenario_name}** was identified as the most optimal modeling approach.")

# Display the primary metrics for the best scenario
col1, col2, col3 = st.columns(3)
col1.metric("Best Scenario", best_scenario_name)
col2.metric("Final Average RMSE", f"{best_rmse:.4f}", help="Lower is better.")
col3.metric("Final Average R² Score", f"{best_r2:.4f}", help="Higher (closer to 1) is better.")

# Display the final features used
st.info(f"**Final Features Used by Model:** `{', '.join(final_features_list)}`", icon="✅")

st.divider()

# --- IMPLICATIONS & LIMITATIONS ---
col_implications, col_limitations = st.columns(2)

with col_implications:
    with st.container(border=True):
        st.subheader("💡 Implications & Benefits")
        st.markdown("""
        - **Early Warning System:** The developed model can serve as a basis for an early warning system for air quality in the relevant area.
        - **Policy Insights:** Feature analysis provides policymakers with insights into which pollutants are most influential and should be prioritized.
        - **Academic Contribution:** The comprehensive preprocessing methodology quantitatively demonstrates that feature selection and outlier removal can significantly improve model accuracy.
        """)

with col_limitations:
    with st.container(border=True):
        st.subheader("⚠️ Model Limitations")
        st.markdown("""
        - **Single-City Data:** The model was trained exclusively on data from Beijing, limiting its generalizability to other cities.
        - **Unforeseen Events:** The ARIMAX model may struggle to predict extreme pollution spikes caused by unexpected events (e.g., large wildfires).
        - **Linearity Assumption:** The underlying ARIMA model assumes linear relationships, whereas real-world pollution dynamics can be highly non-linear.
        """)

# --- FUTURE DEVELOPMENT ---
st.header("🚀 Future Development Directions")
with st.expander("Click to see ideas for future work"):
    st.markdown("""
    1.  **Advanced Models:** Implement deep learning models like **LSTM (Long Short-Term Memory)** or **Transformers** to better capture non-linear patterns.
    2.  **Additional Data Sources:** Integrate data from satellite imagery (Aerosol Optical Depth), traffic data, or industrial activity to enrich the exogenous features.
    3.  **Real-time Deployment:** Develop the system into a real-time application that automatically fetches the latest data and provides hourly forecasts.
    4.  **Multi-City Analysis:** Expand the analysis to compare and model air quality across multiple cities to discover more generalizable patterns.
    """)