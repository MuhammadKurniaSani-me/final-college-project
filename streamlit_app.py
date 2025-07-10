import streamlit as st

st.set_page_config(layout="wide")

st.title('Time Series Analysis with Statistical Significance, Clustering, ARIMA')

with st.expander('About this app'):
  app_description = """This app contains analysis process for my final college project (Skripsi) focusing on time series analysis using an ARIMA model, with data preprocessing techniques, including statistical feature selection and cluster-based outlier detection. The primary goal of this project is to build a robust model for a given time series dataset (e.g., PM2.5 air quality)."""
    # 
  st.write(app_description)
  st.link_button("Check My GitHub", "https://github.com/MuhammadKurniaSani-me/final-college-project")

if st.checkbox("Enable CSS hacks", True):
    
    titleFontSize = "40px"
    titleFontWeight = "500"
    headerFontSize = "32px"
    headerFontWeight = "500"
    subheaderFontSize = "24px"
    subheaderFontWeight = "500"
    
    pageHoverBackgroundColor = "#deddd1"
    pageFontSize = "14px"
    
    activePageBackgroundColor = "#deddd1"
    activePageHoverBackgroundColor = "#deddd1"
    
    
    st.html(
        f"""
        <style>
        body {{
            -webkit-font-smoothing: antialiased;
        }}
        
        h1 {{
            font-size: {titleFontSize} !important;
            font-weight: {titleFontWeight} !important;
        }}
        
        h2 {{
            font-size: {headerFontSize} !important;
            font-weight: {headerFontWeight} !important;
        }}
        
        h3 {{
            font-size: {subheaderFontSize} !important;
            font-weight: {subheaderFontWeight} !important;
        }}
        
        /* Active page in sidebar nav */
        [data-testid="stSidebarNav"] li a[aria-current="page"] {{
            background-color: {activePageBackgroundColor} !important;
        }}
        [data-testid="stSidebarNav"] li a[aria-current="page"]:hover {{
            background-color: {activePageHoverBackgroundColor} !important;
        }}
        
        /* Other pages in sidebar nav */
        [data-testid="stSidebarNav"] li a:hover {{
            background-color: {pageHoverBackgroundColor} !important;
        }}
        [data-testid="stSidebarNav"] li a span {{
            font-size: {pageFontSize} !important;
        }}
        </style>
        """
    )

st.markdown("   ")

pg = st.navigation(
    {
        "General": [
            st.Page("./pages/1_introduction.py", title="Introduction", icon=":material/home:"),
            st.Page("./pages/2_data_overview.py", title="Data Overview", icon=":material/table_chart:"),
            st.Page("./pages/3_data_preprocessing.py", title="Preprocessing", icon=":material/tune:"),
            st.Page("./pages/4_statistical_significance.py", title="Statistical Significance", icon=":material/functions:"),
            st.Page("./pages/5_outlier_removal.py", title="Outlier Removal", icon=":material/filter_alt_off:"),
            st.Page("./pages/6_evaluation.py", title="Scenario Evaluation", icon=":material/timeline:"),
            st.Page("./pages/7_prediction.py", title="PM 2.5 Prediction", icon=":material/timeline:"),
            st.Page("./pages/8_conlusion.py", title="Conclusion", icon=":material/flag:"),
            st.Page("./pages/9_scenario_4_evaluation.py", title="Scenario 4 Evaluation", icon=":material/filter_4:"),
            st.Page("./pages/10_scenario_3_evaluation.py", title="Scenario 3 Evaluation", icon=":material/filter_3:"),
            st.Page("./pages/11_scenario_2_evaluation.py", title="Scenario 2 Evaluation", icon=":material/filter_2:"),
            st.Page("./pages/12_scenario_1_evaluation.py", title="Scenario 1 Evaluation", icon=":material/filter_1:"),
        ],
        # "Admin": [st.Page(page3, title="Settings", icon=":material/settings:")],
    }
)


pg.run()


