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
STATION_NAMES = [
    'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi',
    'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan',
    'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'
]

ALL_FEATURE_DESCRIPTIONS = {
    "YEAR": "Tahun dari data.",
    "MONTH": "Bulan dari data.",
    "DAY": "Hari dari data.",
    "HOUR": "Jam dari data.",
    "PM2.5": "Jumlah konsentrasi partikel ukuran 2.5 mikrometer (µg/m³).",
    "PM10": "Jumlah konsentrasi partikel ukuran 10 mikrometer (µg/m³).",
    "SO2": "Jumlah konsentrasi belerang dioksida (µg/m³).",
    "NO2": "Jumlah konsentrasi nitrogen dioksida (µg/m³).",
    "CO": "Jumlah konsentrasi karbon monoksida (µg/m³).",
    "O3": "Jumlah konsentrasi ozon (µg/m³).",
    "TEMP": "Nilai suhu (°C).",
    "PRES": "Nilai tekanan atmosfer (hPa).",
    "DEWP": "Suhu titik embun (°C).",
    "RAIN": "Curah hujan (mm).",
    "WD": "Arah mata angin.",
    "WSPM": "Kecepatan angin (m/s).",
    "STATION": "Nama stasiun pemantauan."
}


## define functions
def styled_header(text, emoji, divider_color='grey', level='header'):
    """Creates a styled subheader with an emoji."""

    if level == 'header':
        st.header(f"{emoji} {text}", divider=divider_color)

    if level == 'subheader':
        st.subheader(f"{emoji} {text}", divider=divider_color)


def introduction_section():
    """"
    """
    
    styled_header('Get Along with Data', '👇', divider_color="grey", level='header')
    st.markdown("Data kualitas udara kota Beijing China yang sudah banyak dipakai serta tersedia di _UC Irvine Machine Learning Repository_")
    st.page_link("https://doi.org/10.1098/rspa.2017.0457", label="Zhang et al. 2017", icon=":material/open_in_new:")
    st.markdown("   ")


@st.cache_data
def fetch_and_clean_data(station_name):
    """
    """

    # url_path = 'https://raw.githubusercontent.com/MuhammadKurniaSani-me/datasets/refs/heads/main/real-dataset/beijing-multi-site-air-quality-data/'
    file_path = "./datas/locations/"
    data = pd.read_csv(f'{file_path}PRSA_Data_{station_name}_20130301-20170228.csv')

    return data


@st.cache_data
def plot_missing_values_bar(df):
    """
    Visualizes missing values using an enhanced Seaborn bar chart.
    """

    # Calculate missing values and filter out columns with no missing data
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    # If there are no missing values, return a message
    if missing_values.empty:
        st.success("Congratulations! No missing values found in the dataset.")
        return None

    # Set plot style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create a color palette
    palette = sns.color_palette("viridis", len(missing_values))

    # Create bar plot
    barplot = sns.barplot(
        x=missing_values.index,
        y=missing_values.values,
        ax=ax,
        palette=palette
    )

    # Add data labels on top of each bar
    for p in barplot.patches:
        ax.annotate(
            format(p.get_height(), '.0f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points',
            fontsize=10
        )

    # Set titles and labels with better formatting
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Number of Missing Values', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Adjust layout to prevent labels from being cut off 
    fig.tight_layout()

    st.write("🎯 Number of missing values in each column")
    st.page_link("https://doi.org/10.48550/arXiv.2410.03712", label="S. Alsufyani et al. 2024", icon=":material/open_in_new:")

    return fig


@st.cache_data
def plot_correlations(df):
    """
    Visualizes the correlation of missing values using an enhanced Seaborn heatmap.
    This version is cleaner and designed for Streamlit.
    """

    le = LabelEncoder()
    correlation_df = df.copy(deep=True)
    correlation_df['station'] = le.fit_transform(correlation_df.loc[:, 'station'])
    correlation_df['wd'] = le.fit_transform(correlation_df.loc[:, 'wd'])
    independent_features = correlation_df.iloc[:, 4:].columns
    correlation_matrix = correlation_df.loc[:, independent_features].corr()

    # Create a mask to hide the upper triangle (since the matrix is symmetric)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,          # Show the correlation values
        fmt=".2f",           # Format values to 2 decimal places
        cmap='coolwarm',     # Use a diverging colormap
        center=0,
        linewidths=.5,
        cbar_kws={"shrink": .75}
    )

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)
    fig.tight_layout()

    st.write("🎯 Correlation influence of each feature")
    st.page_link("https://doi.org/10.1186/s40537-021-00548-1", label="A. Bekkar et al. 2021", icon=":material/open_in_new:")

    return fig


# @st.cache_data
# def show_descriptive_stats(df):
#     """
#     Displays descriptive statistics for the numerical columns.
#     """

#     styled_subheader("Descriptive Statistics", "📊")
#     st.write("Here's a statistical summary of the numerical features in the dataset. This helps in understanding the distribution, scale, and central tendency of the data.")
#     st.dataframe(df.describe(), use_container_width=True)


@st.cache_data
def plot_time_series(df, y_col):
    """
    Plots the main dependent variable over time in a cool, interactive plot.
    """

    time_index_col = pd.to_datetime(df[['year','month','day','hour']])

    fig = px.line(
        df,
        x=time_index_col,
        y=y_col,
        title=f' ',
        template='plotly_dark',
        markers=True
    )

    # Enhance the plot aesthetics
    fig.update_traces(
        line=dict(width=3, color='#33CFA5'),
        marker=dict(size=8, color='#33CFA5', symbol='circle'),
        fill='tozeroy',  # Adds a cool shaded area under the line
        fillcolor='rgba(51, 207, 165, 0.2)'
    )
    fig.update_layout(
        xaxis_title='Time Index',
        yaxis_title=f'Normalized {y_col}',
        title_font_size=20,
        title_x=0.5,
        hovermode='x unified'
    )

    st.write(f'🎯 Value of {y_col} over time')
    st.page_link("https://doi.org/10.1186/s40537-021-00548-1", label="A. Bekkar et al. 2021", icon=":material/open_in_new:")

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_feature_distributions(df):
    """
    Plots histograms for all numerical features to show their distributions.
    """

    # styled_header("Feature Distribution Insights", "👇", divider_color='grey', level='subheader')
    st.write("🎯 The distribution of each feature with histograms")
    st.page_link("https://doi.org/10.1098/rspa.2017.0457", label="Zhang et al. 2017", icon=":material/open_in_new:")

    numeric_columns = df.select_dtypes(include=['number']).columns
    cols = st.columns(2) # Create 2 columns for a cleaner layout

    for i, col_name in enumerate(numeric_columns):
        with cols[i % 2]:
            fig = px.histogram(
                df,
                x=col_name,
                title=f'Distribution of {col_name}',
                template='plotly_dark',
                color_discrete_sequence=['#FF6347'] # Tomato color
            )
            fig.update_layout(
                bargap=0.1,
                xaxis_title=f'Normalized {col_name}',
                yaxis_title='Frequency'
            )
            st.plotly_chart(fig, use_container_width=True)


def plot_scatter_relationships(df, x_col, y_cols):
    """
    Creates interactive scatter plots to visualize the relationship
    between the dependent variable and key independent variables.
    """

    # styled_header(f"Relationships with {x_col}", "👇", divider_color='grey', level='subheader')
    st.write(f"🎯 The Scatter plots help us visually confirm the relationships suggested by the correlation matrix")
    st.page_link("https://doi.org/10.1098/rspa.2017.0457", label="Zhang et al. 2017", icon=":material/open_in_new:")

    cols = st.columns(2) # Create 2 columns for layout

    for i, y_col in enumerate(y_cols):
        with cols[i % 2]:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f'{x_col} vs. {y_col}',
                template='plotly_dark',
                trendline="ols", # Adds an Ordinary Least Squares regression line
                trendline_color_override="red",
                color_discrete_sequence=['#1E90FF'] # DodgerBlue
            )
            fig.update_layout(
                xaxis_title=f'Normalized {x_col}',
                yaxis_title=f'Normalized {y_col}'
            )
            st.plotly_chart(fig, use_container_width=True)


## display contents

### opening word 
introduction_section()

### load data
selected_station = st.selectbox("Location options", STATION_NAMES)
st.write('You selected location in:', selected_station)
df = fetch_and_clean_data(selected_station).iloc[:, 1:]

if 'dataframe' not in st.session_state:
    st.session_state['main_dataframe'] = df
    st.session_state['data_location'] = selected_station
    st.session_state['locations_name'] = STATION_NAMES

#### share df to all pages
st.dataframe(df)

### describe feature
styled_header('What is inside of the dataset?', '👇', divider_color='grey', level='subheader')
st.write("🎯 Features description")
st.page_link("https://doi.org/10.1002/env.2819", label="Zhang et al. 2023", icon=":material/open_in_new:")
filtered_descriptions = [(feature, ALL_FEATURE_DESCRIPTIONS.get(str.upper(feature), "Deskripsi tidak ditemukan.")) for feature in df.columns]
df_features = pd.DataFrame(filtered_descriptions, columns=['Feature', 'Deskripsi'])
st.dataframe(df_features, use_container_width=True)

### explore data
st.markdown("   ")
st.pyplot(plot_missing_values_bar(df))
st.markdown("   ")
st.pyplot(plot_correlations(df)) 
st.markdown("   ")
plot_time_series(df, y_col='PM2.5')
st.markdown("   ")
plot_feature_distributions(df)
st.markdown("   ")
plot_scatter_relationships(df,'PM2.5', ['PM10', 'CO', 'NO2', 'SO2'])
st.markdown("   ")