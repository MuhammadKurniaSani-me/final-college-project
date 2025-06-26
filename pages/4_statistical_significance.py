## define modules
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


## define constants
FIRST_ITEM = 0
LAST_ITEM = -1
ONE_STEP = 1
MISSING_PERCENTAGE = 0.1
TARGET_COLUMN = 'PM2.5'


## define functions
def styled_header(text, emoji, divider_color='grey', level='header', additional_information=False):
    """Creates a styled subheader with an emoji."""

    if level == 'header':
        st.header(f"{emoji} {text}", divider=divider_color)

    if level == 'subheader':
        st.subheader(f"{emoji} {text}", divider=divider_color)

    if additional_information:
        st.write(additional_information)


def introduction_section():
    """"
    """
    
    styled_header("Let's statistic do the magic!", '👇', divider_color="grey", level='header')
    st.markdown("Select the most possible significance features, with Pearson Correlation & Variance Inflation Factor.")
    st.page_link("https://doi.org/10.1016/j.heliyon.2023.e13483", label="Ilu et al. 2023", icon=":material/open_in_new:")
    st.markdown("   ")


def method_explanation():
    """
    """

    # --- KEY TAKEAWAYS (4 COLUMNS) ---
    col1, col2 = st.columns(2)

    # --- COLUMN 1: OpenAI Dominance ---
    with col1:
        st.markdown("### Pearson Correlation")
        st.write(
            "Big city like Beijing in China has become on of the many city with high pollution air"
        )
        if st.button("Top 5 big cities", key="b1"):
            st.info("Redirecting to top models...")

    # --- COLUMN 2: Multi-Agent Future ---
    with col2:
        st.markdown("### Variance Inflation Factor")
        st.write(
            "Time series data increases its volume and time point with some external factor that influence it"
        )
        if st.button("See more data", key="b2"):
            st.info("Redirecting to orchestration tools...")

    st.markdown("   ") 



## Shows contents
introduction_section()
method_explanation()

# this is comment to check if git push again is error or not