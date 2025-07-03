# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA # Untuk visualisasi
import plotly.graph_objects as go
import scipy.spatial.distance
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import warnings


# --- KONSTANTA ---
ALL_FEATURE_DESCRIPTIONS = {
    "YEAR": "Tahun dari data.", "MONTH": "Bulan dari data.", "DAY": "Hari dari data.", "HOUR": "Jam dari data.",
    "PM2.5": "Konsentrasi partikel PM2.5 (µg/m³).", "PM10": "Konsentrasi partikel PM10 (µg/m³).",
    "SO2": "Konsentrasi SO₂ (µg/m³).", "NO2": "Konsentrasi NO₂ (µg/m³).", "CO": "Konsentrasi CO (µg/m³).",
    "O3": "Konsentrasi O₃ (µg/m³).", "TEMP": "Suhu (°C).", "PRES": "Tekanan atmosfer (hPa).",
    "DEWP": "Suhu titik embun (°C).", "RAIN": "Curah hujan (mm).", "WD": "Arah mata angin.",
    "WSPM": "Kecepatan angin (m/s).", "STATION": "Nama stasiun pemantauan."
}

# --- KONSTANTA PREPROCESSING ---
WIND_DIRECTION_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135,
    'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
    'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}


# --- FUNGSI PEMUATAN DATA ---
# @st.cache_data
# def load_station_data(station_name):
#     """Memuat dan membersihkan data untuk stasiun yang dipilih."""
#     try:
#         # Ganti dengan path yang sesuai jika perlu
#         file_path = f'./datas/locations/PRSA_Data_{station_name}_20130301-20170228.csv'
#         df = pd.read_csv(file_path)
#         # Menghapus kolom 'No' yang tidak diperlukan
#         df = df.iloc[:, 1:]
#         return df
#     except FileNotFoundError:
#         st.error(f"File data untuk stasiun {station_name} tidak ditemukan.")
#         return None

# @st.cache_data
# def load_station_data(station_name):
#     """
#     Memuat data untuk stasiun yang dipilih, membuat DatetimeIndex,
#     dan menghapus kolom waktu yang asli.
#     """
#     try:
#         file_path = f'./datas/locations/PRSA_Data_{station_name}_20130301-20170228.csv'
#         df = pd.read_csv(file_path)
#         df = df.iloc[:, 1:] # Hapus kolom 'No'

#         # --- PERUBAHAN UTAMA DI SINI ---
#         # 1. Gabungkan kolom waktu menjadi satu kolom 'datetime'
#         df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
#         # 2. Jadikan kolom 'datetime' sebagai indeks baru DataFrame
#         df.set_index('datetime', inplace=True)
        
#         # 3. Hapus kolom waktu asli karena sudah tidak diperlukan
#         df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
#         # --- AKHIR PERUBAHAN ---
        
#         return df
        
#     except FileNotFoundError:
#         st.error(f"File data untuk stasiun {station_name} tidak ditemukan.")
#         return None

# utils.py
# Ganti fungsi load_station_data Anda dengan versi yang sudah benar ini

@st.cache_data
def load_station_data(station_name):
    """
    Memuat data, membuat DatetimeIndex, dan menghapus kolom waktu asli.
    Ini adalah satu-satunya sumber data mentah untuk memastikan konsistensi.
    """
    try:
        url_path = url_path = 'https://raw.githubusercontent.com/MuhammadKurniaSani-me/datasets/refs/heads/main/real-dataset/beijing-multi-site-air-quality-data/'
        file_path = f'{url_path}PRSA_Data_{station_name}_20130301-20170228.csv'
        # file_path = f'./datas/locations/PRSA_Data_{station_name}_20130301-20170228.csv'
        df = pd.read_csv(file_path)
        df = df.iloc[:, 1:] # Hapus kolom 'No'

        # 1. Gabungkan kolom waktu menjadi satu kolom 'datetime'
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # 2. Jadikan kolom 'datetime' sebagai indeks baru DataFrame
        df.set_index('datetime', inplace=True)
        
        # 3. Hapus kolom waktu asli karena sudah tidak diperlukan lagi
        df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
        
        return df
        
    except FileNotFoundError:
        st.error(f"File data untuk stasiun {station_name} tidak ditemukan.")
        return None

# --- FUNGSI PLOTTING (HANYA MENGEMBALIKAN GAMBAR) ---

def create_missing_values_plot(df):
    """Membuat bar plot untuk nilai yang hilang."""
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if missing_values.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette("viridis", len(missing_values))
    barplot = sns.barplot(x=missing_values.index, y=missing_values.values, ax=ax, palette=palette)
    for p in barplot.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
    ax.set_title('Jumlah Nilai yang Hilang per Kolom', fontsize=16)
    ax.set_xlabel('Kolom', fontsize=12)
    ax.set_ylabel('Jumlah Nilai yang Hilang', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig

def create_correlation_plot(df):
    """
    Membuat heatmap korelasi, secara otomatis mengecualikan fitur waktu.
    """
    # --- PERUBAHAN DI SINI ---
    # Definisikan fitur waktu yang akan dikecualikan dari analisis korelasi
    time_features = ['year', 'month', 'day', 'hour']
    
    # Buat salinan DataFrame dan hapus fitur waktu
    # errors='ignore' akan mencegah error jika kolom tersebut sudah tidak ada
    df_for_corr = df.drop(columns=time_features, errors='ignore')
    # --- AKHIR PERUBAHAN ---

    # Sisa logika sekarang berjalan pada DataFrame yang sudah difilter
    le = LabelEncoder()
    correlation_df = df_for_corr.copy(deep=True)
    
    # Encode kolom kategorikal hanya untuk korelasi
    if 'wd' in correlation_df.columns and correlation_df['wd'].dtype == 'object':
        correlation_df['wd'] = le.fit_transform(correlation_df['wd'])
    if 'station' in correlation_df.columns and correlation_df['station'].dtype == 'object':
         correlation_df['station'] = le.fit_transform(correlation_df['station'])
    
    numeric_df = correlation_df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10)) # Ukuran disesuaikan agar tidak terlalu besar
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax, annot_kws={"size": 8}) # Ukuran font anotasi disesuaikan
    ax.set_title('Heatmap Korelasi Antar Fitur Non-Waktu', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig

def create_timeseries_plot(df, y_col):
    """Membuat plot runtun waktu interaktif."""
    df_plot = df.copy()
    fig = px.line(df_plot, x=df_plot.index, y=y_col, title=f'Tren Nilai {y_col} dari Waktu ke Waktu')
    fig.update_traces(line=dict(width=2, color='#1E90FF'))
    fig.update_layout(xaxis_title='Tanggal', yaxis_title=f'Nilai {y_col}', hovermode='x unified')
    return fig

def create_distribution_plot(df, column):
    """Membuat histogram untuk distribusi satu fitur."""
    fig = px.histogram(df, x=column, title=f'Distribusi Fitur {column}', nbins=50,
                       color_discrete_sequence=['#FF6347'])
    fig.update_layout(bargap=0.1, xaxis_title='Nilai', yaxis_title='Frekuensi')
    return fig

    """Membuat scatter plot untuk melihat hubungan dua fitur."""
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Hubungan antara {x_col} dan {y_col}',
                     trendline="ols", trendline_color_override="red")
    return fig


# --- FUNGSI PREPROCESSING ---

def preprocess_encoding(df, station_name, station_code_map):
    """Meng-encode kolom 'wd' dan 'station'."""
    df_encoded = df.copy()
    if 'wd' in df_encoded.columns:
        df_encoded['wd'] = df_encoded['wd'].map(WIND_DIRECTION_MAP)
    # Kolom 'station' sudah ada dalam data mentah, kita hanya perlu memastikan ada
    # Jika tidak ada, kita tambahkan berdasarkan nama stasiun yang dipilih
    if 'station' not in df_encoded.columns:
        df_encoded['station'] = station_name
    
    if 'station' in df_encoded.columns and df_encoded['station'].dtype == 'object':
         df_encoded['station'] = df_encoded['station'].map(station_code_map)
            
    return df_encoded

def preprocess_impute(df):
    """Mengisi nilai yang hilang menggunakan interpolasi linear."""
    # Pandas' interpolate function is highly optimized, fast, and reliable.
    # It replaces all the custom linear imputation logic in one line.
    df_imputed = df.interpolate(method='linear', limit_direction='both', axis=0)
    return df_imputed

def preprocess_scale(df):
    """Menskalakan data menggunakan MinMaxScaler dari Scikit-learn."""
    # MinMaxScaler is the standard and most efficient way to perform this normalization.
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    return df_scaled, scaler


# --- FUNGSI STYLING (SUDAH ADA/DIPERBARUI) ---

def highlight_encoded_and_na(df, cols_to_highlight):
    """Menyorot kolom yang di-encode dan nilai NaN."""
    df_styled = df.copy()
    nan_style_color = '#FFCDD2' # Light Red
    encoded_header_props = [('background-color', '#C8E6C9')]
    
    header_rules = [
        {'selector': f'th.col_heading.col{df.columns.get_loc(col) + 1}', 'props': encoded_header_props}
        for col in cols_to_highlight if col in df.columns
    ]
    
    styler = df_styled.style.set_properties(
        subset=cols_to_highlight, **{'background-color': '#E8F5E9'}
    ).set_table_styles(header_rules).highlight_null()
    
    return styler

def highlight_min_max(df, precision=2):
    """Menyorot nilai min/max di setiap kolom."""
    min_color = '#FFCDD2'
    max_color = '#C8E6C9'
    format_string = f"{{:.{precision}f}}"
    
    styler = df.style.highlight_max(axis=0, color=max_color).highlight_min(axis=0, color=min_color).format(format_string, na_rep="-")
    return styler


def calculate_vif(df):
    """
    Menghitung Variance Inflation Factor (VIF) untuk setiap fitur.
    """
    # Hanya pilih kolom numerik dan hapus kolom dengan varians nol
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
    
    vif_data = pd.DataFrame()
    vif_data["Fitur"] = numeric_df.columns
    
    # Hitung VIF untuk setiap fitur
    vif_data["Skor VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                           for i in range(len(numeric_df.columns))]
    
    return vif_data.sort_values(by="Skor VIF", ascending=False).reset_index(drop=True)


# --- FUNGSI BARU: K-MEANS DENGAN MAHALANOBIS DISTANCE ---
def kmeans_mahalanobis(df, k, max_iterations=100):
    """
    Menjalankan algoritma K-Means menggunakan Mahalanobis distance.
    
    Returns:
    --------
    labels (np.array): Label klaster untuk setiap titik data.
    centroids (np.array): Pusat klaster final.
    inertia (float): Sum of squared Mahalanobis distances to closest centroid.
    """
    data = df.to_numpy()
    n_samples, n_features = data.shape
    
    # 1. Hitung inverse dari covariance matrix sekali saja
    try:
        cov_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to identity matrix if covariance is singular (no correlation)
        inv_cov_matrix = np.identity(n_features)

    # 2. Inisialisasi centroid menggunakan metode K-Means++ dari Scikit-learn (cara cerdas)
    kmeans_init = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    kmeans_init.fit(data)
    centroids = kmeans_init.cluster_centers_

    # 3. Iterasi untuk assignment dan update
    for _ in range(max_iterations):
        # Assignment Step: Hitung Mahalanobis distance untuk setiap titik ke setiap centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            # Gunakan scipy.spatial.distance.mahalanobis
            distances[:, i] = [scipy.spatial.distance.mahalanobis(point, centroids[i], inv_cov_matrix) for point in data]
        
        # Tentukan klaster berdasarkan jarak terdekat
        labels = np.argmin(distances, axis=1)
        
        # Update Step: Hitung ulang centroid baru
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Cek konvergensi
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Hitung inertia (Sum of Squared Mahalanobis Distances)
    final_distances = np.min(distances, axis=1)
    inertia = np.sum(final_distances**2)
    
    return labels, centroids, inertia

@st.cache_data
def run_clustering_analysis(df, max_k=10):
    """
    Menjalankan K-Means Mahalanobis untuk berbagai nilai k untuk menemukan k optimal.
    """
    sse_scores = {}
    silhouette_scores = {}
    
    for k in range(2, max_k + 1):
        # Gunakan fungsi kustom kita
        labels, _, sse = kmeans_mahalanobis(df, k)
        sse_scores[k] = sse
        # Silhouette score dihitung dengan Euclidean karena itu standarnya,
        # tapi berdasarkan label dari Mahalanobis-KMeans. Ini pendekatan umum.
        silhouette_scores[k] = silhouette_score(df, labels, metric='mahalanobis')
        
    return sse_scores, silhouette_scores

def create_elbow_plot(sse_scores):
    """Membuat plot untuk Elbow Method."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(sse_scores.keys()), y=list(sse_scores.values()),
                             mode='lines+markers', marker=dict(color='#1E90FF')))
    fig.update_layout(title="Elbow Method untuk Menentukan k Optimal",
                      xaxis_title="Jumlah Klaster (k)",
                      yaxis_title="Sum of Squared Errors (SSE)")
    return fig

def create_silhouette_plot(silhouette_scores):
    """Membuat plot untuk Silhouette Score."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()),
                             mode='lines+markers', marker=dict(color='#FF6347')))
    fig.update_layout(title="Rata-rata Silhouette Score untuk Setiap k",
                      xaxis_title="Jumlah Klaster (k)",
                      yaxis_title="Rata-rata Silhouette Score")
    return fig

# def get_outliers_df(df, k, silhouette_threshold=0.0):
#     """
#     Menjalankan K-Means dengan k optimal dan mengidentifikasi outlier.
#     """
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
#     df_result = df.copy()
    
#     # Tambahkan label klaster dan silhouette score ke DataFrame
#     df_result['cluster'] = kmeans.fit_predict(df_result)
#     df_result['silhouette_score'] = silhouette_samples(df_result.drop('cluster', axis=1), df_result['cluster'])
    
#     # Pisahkan antara inlier dan outlier
#     inliers = df_result[df_result['silhouette_score'] >= silhouette_threshold].copy()
#     outliers = df_result[df_result['silhouette_score'] < silhouette_threshold].copy()
    
#     # Hapus kolom bantuan dari DataFrame final
#     inliers.drop(columns=['cluster', 'silhouette_score'], inplace=True)
    
#     return inliers, outliers, df_result

def get_outliers_df(df, k, silhouette_threshold=0.0):
    """
    Menjalankan K-Means dengan Mahalanobis Distance dan mengidentifikasi outlier.
    """
    df_result = df.copy()
    
    # --- PERUBAHAN DI SINI ---
    # Hapus pemanggilan KMeans standar dari Scikit-learn
    # kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    # df_result['cluster'] = kmeans.fit_predict(df_result)
    
    # Panggil fungsi kustom kmeans_mahalanobis untuk mendapatkan label klaster
    # Kita hanya butuh 'labels', jadi kita abaikan 'centroids' dan 'inertia' dengan '_'
    labels, _, _ = kmeans_mahalanobis(df_result, k)
    df_result['cluster'] = labels
    # --- AKHIR PERUBAHAN ---
    
    # Sisa logika tetap sama, karena hanya bergantung pada label klaster
    df_result['silhouette_score'] = silhouette_samples(df_result.drop('cluster', axis=1), df_result['cluster'])
    
    # Pisahkan antara inlier dan outlier
    inliers = df_result[df_result['silhouette_score'] >= silhouette_threshold].copy()
    outliers = df_result[df_result['silhouette_score'] < silhouette_threshold].copy()
    
    # Hapus kolom bantuan dari DataFrame final
    inliers.drop(columns=['cluster', 'silhouette_score'], inplace=True)
    
    return inliers, outliers, df_result

def create_cluster_visualization(df_with_clusters):
    """
    Membuat scatter plot 2D dari klaster menggunakan PCA.
    """
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_with_clusters.drop(['cluster', 'silhouette_score'], axis=1))
    
    df_plot = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
    df_plot['cluster'] = df_with_clusters['cluster'].values
    df_plot['silhouette_score'] = df_with_clusters['silhouette_score'].values
    
    fig = px.scatter(df_plot, x='PC1', y='PC2', color='cluster',
                     title="Visualisasi Klaster (reduksi dimensi dengan PCA)",
                     hover_data=['silhouette_score'],
                     color_continuous_scale=px.colors.qualitative.Vivid)
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return fig


# --- FUNGSI EVALUASI MODEL ---

def find_best_order_grid_search(series, max_p=3, max_q=3):
    """
    Menemukan order ARIMA (p,d,q) terbaik menggunakan grid search dan uji ADF.
    
    Returns:
    --------
    tuple: Order (p,d,q) terbaik yang ditemukan.
    """
    # 1. Menentukan orde diferensiasi (d) dengan Uji ADF
    adf_test = adfuller(series, autolag='AIC')
    p_value = adf_test[1]
    
    d = 0
    if p_value > 0.05:
        # Jika data tidak stasioner, lakukan diferensiasi sekali
        d = 1
        diff_series = series.diff().dropna()
        adf_test_diff = adfuller(diff_series, autolag='AIC')
        # (Untuk kesederhanaan, kita asumsikan d=1 sudah cukup, bisa diperluas jika perlu)

    # 2. Grid Search untuk p dan q
    best_aic = float('inf')
    best_order = None
    
    p_range = range(max_p + 1)
    q_range = range(max_q + 1)
    
    # Matikan warning dari statsmodels yang sering muncul saat model tidak konvergen
    warnings.filterwarnings("ignore") 
    
    for p in p_range:
        for q in q_range:
            # Lewati order (0,d,0) karena itu hanya random walk
            if p == 0 and q == 0:
                continue
            
            try:
                model = SARIMAX(series, order=(p, d, q))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except Exception as e:
                # Lewati kombinasi yang menyebabkan error
                continue
                
    warnings.filterwarnings("default") # Kembalikan warning seperti semula
    
    # Jika tidak ada order yang ditemukan (sangat jarang), kembalikan default
    if best_order is None:
        return (1, d, 1)
        
    return best_order

def rep_holdout_splitter(data, n_repetitions=10, train_percent=0.8, test_percent=0.1):
    """
    Generator function to yield train/test splits using a sliding window approach.
    This more accurately reflects the repetition hold-out methodology.
    """
    N = len(data)
    train_size = math.floor(train_percent * N)
    test_size = math.floor(test_percent * N)
    window_size = train_size + test_size

    if window_size > N:
        raise ValueError(f"Total window size ({window_size}) cannot be larger than the dataset size ({N}).")

    if n_repetitions < 1:
        raise ValueError("Number of repetitions must be at least 1.")

    # Calculate the total "room" available for the window to slide
    max_start_offset = N - window_size
    
    # Calculate the step size to slide the window for each repetition
    # If only 1 repetition, step is 0. Otherwise, distribute the slide across repetitions.
    slide_step = 0
    if n_repetitions > 1:
        # Ensure we don't try to divide by zero if max_start_offset is negative or zero
        if max_start_offset > 0:
            slide_step = max_start_offset // (n_repetitions - 1)
        else:
            # Not enough data to slide, all repetitions will use the same window
            n_repetitions = 1 # Force to 1 repetition
            st.warning(f"Dataset tidak cukup besar untuk {n_repetitions} jendela geser yang berbeda. Hanya 1 repetisi yang akan dijalankan.")


    print(f"Running splitter: N={N}, train={train_size}, test={test_size}, reps={n_repetitions}, step={slide_step}")

    for i in range(n_repetitions):
        start_index = i * slide_step
        train_end_index = start_index + train_size
        test_end_index = train_end_index + test_size
        
        # Yield the train and test splits for the current window
        train_split = data.iloc[start_index:train_end_index]
        test_split = data.iloc[train_end_index:test_end_index]

        # Final safety check to ensure splits are not empty
        if train_split.empty or test_split.empty:
            st.error("Gagal membuat potongan data. Periksa kembali parameter train/test percent.")
            break

        yield train_split, test_split


# utils.py

@st.cache_data
def run_arimax_evaluation(
    df, scenario_name, use_feature_selection, use_outlier_removal, 
    _session_state, _status_container=None # <-- UBAH NAMA ARGUMEN DI SINI
):
    """
    Versi ini menggunakan _status_container yang diabaikan oleh cache.
    """
    
    target_variable = 'PM2.5'
    all_rmse_scores = []
    all_r2_scores = []
    representative_run_data = {}

    if use_feature_selection:
        features = _session_state.final_features
    else:
        all_cols = _session_state.df_imputed.columns.tolist()
        time_features = ['year', 'month', 'day', 'hour']
        features = [col for col in all_cols if col not in time_features and col != target_variable]

    df_scenario = df[[target_variable] + features].copy()

    splits = list(rep_holdout_splitter(df_scenario, n_repetitions=10, train_percent=0.8, test_percent=0.1))

    if not splits:
        st.error("Gagal membuat potongan data (splits) untuk evaluasi. Periksa parameter atau ukuran data.")
        return None # Hentikan eksekusi jika tidak ada split

    for i, (train_df, test_df) in enumerate(splits):
        status_message = f"Repetisi {i + 1}/{len(splits)}: "
        
        # --- GUNAKAN NAMA BARU DI SINI ---
        if _status_container:
            _status_container.update(label=status_message + "Mencari order ARIMA terbaik...")
        
        order = find_best_order_grid_search(train_df[target_variable], max_p=3, max_q=3)

        # --- DAN DI SINI ---
        if _status_container:
            _status_container.update(label=status_message + f"Melatih model ARIMAX{order}...")
        
        train_df_final = train_df.copy()
        if use_outlier_removal:
            train_df_final, _, _ = get_outliers_df(train_df, k=2) 
        
        endog_train = train_df_final[target_variable]
        exog_train = train_df_final[features]
        endog_test = test_df[target_variable]
        exog_test = test_df[features]

        order = find_best_order_grid_search(endog_train, max_p=3, max_q=3)
        model = SARIMAX(endog_train, exog=exog_train, order=order)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=len(endog_test), exog=exog_test)
        forecast_mean = forecast.predicted_mean
        
        # --- PERUBAHAN DI SINI ---
        # Hitung RMSE
        rmse = np.sqrt(mean_squared_error(endog_test, forecast_mean))
        all_rmse_scores.append(rmse)
        
        # Hitung R-squared
        r2 = r2_score(endog_test, forecast_mean)
        all_r2_scores.append(r2)
        # --- AKHIR PERUBAHAN ---

        if i == 0:
            representative_run_data = {
                'train_actual': endog_train, 'test_actual': endog_test,
                'test_predicted': forecast_mean, 'order': order
            }

    # ... (return dictionary tetap sama) ...
    avg_rmse = np.mean(all_rmse_scores)
    std_rmse = np.std(all_rmse_scores)
    avg_r2 = np.mean(all_r2_scores)
    std_r2 = np.std(all_r2_scores)
    
    # --- PASTIKAN BAGIAN RETURN ANDA SEPERTI INI ---
    # Ini adalah sebuah DICTIONARY, bukan set.
    return {
        "scenario_name": scenario_name,
        "avg_rmse": avg_rmse,
        "std_rmse": std_rmse,
        "all_scores_rmse": all_rmse_scores,
        "avg_r2": avg_r2,
        "std_r2": std_r2,
        "all_scores_r2": all_r2_scores,
        "plot_data": representative_run_data
    }

def create_evaluation_plot(plot_data):
    """Membuat plot perbandingan antara data aktual dan hasil prediksi."""
    fig = go.Figure()
    
    # Plot data latih
    fig.add_trace(go.Scatter(x=plot_data['train_actual'].index, y=plot_data['train_actual'],
                             mode='lines', name='Data Latih Aktual', line=dict(color='grey')))
    
    # Plot data uji aktual
    fig.add_trace(go.Scatter(x=plot_data['test_actual'].index, y=plot_data['test_actual'],
                             mode='lines', name='Data Uji Aktual', line=dict(color='#1E90FF', width=3)))
    
    # Plot data prediksi
    fig.add_trace(go.Scatter(x=plot_data['test_predicted'].index, y=plot_data['test_predicted'],
                             mode='lines', name='Hasil Prediksi', line=dict(color='#FF6347', dash='dash', width=3)))
    
    fig.update_layout(title=f"Perbandingan Prediksi vs Aktual (ARIMAX{plot_data['order']})",
                      xaxis_title="Indeks Waktu", yaxis_title="Nilai PM2.5 (Ternormalisasi)")
    
    return fig

def find_best_order_grid_search(series, max_p=3, max_q=3):
    """
    Menemukan order ARIMA (p,d,q) terbaik menggunakan grid search dan uji ADF.
    """
    # 1. Menentukan orde diferensiasi (d) dengan Uji ADF
    adf_test = adfuller(series, autolag='AIC')
    p_value = adf_test[1]
    
    d = 0
    if p_value > 0.05:
        d = 1
        
    # 2. Grid Search untuk p dan q
    best_aic = float('inf')
    best_order = None
    
    p_range = range(max_p + 1)
    q_range = range(max_q + 1)
    
    warnings.filterwarnings("ignore") 
    
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                model = SARIMAX(series, order=(p, d, q))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except Exception:
                continue
                
    warnings.filterwarnings("default")
    
    if best_order is None:
        return (1, d, 1) # Fallback
        
    return best_order

def inverse_transform_predictions(scaled_predictions, scaler, all_feature_names, target_variable):
    """
    Mengubah prediksi yang ternormalisasi kembali ke skala aslinya.
    """
    # Buat DataFrame dummy dengan struktur yang sama seperti saat scaler dilatih
    # Ini penting karena scaler mengharapkan jumlah kolom yang sama persis
    dummy_df = pd.DataFrame(np.zeros((len(scaled_predictions), len(all_feature_names))), columns=all_feature_names)
    
    # Masukkan prediksi yang ternormalisasi ke kolom target di DataFrame dummy
    dummy_df[target_variable] = scaled_predictions
    
    # Lakukan inverse transform pada seluruh DataFrame dummy
    inversed_df = scaler.inverse_transform(dummy_df)
    
    # Ambil nilai yang sudah di-inverse HANYA dari kolom target
    # Cari indeks kolom target untuk mengambil data dari array NumPy
    try:
        target_idx = all_feature_names.index(target_variable)
    except ValueError:
        # Fallback jika nama kolom tidak ditemukan, meskipun seharusnya selalu ada
        return np.array([0] * len(scaled_predictions))

    original_scale_predictions = inversed_df[:, target_idx]
    
    return original_scale_predictions

# @st.cache_resource(show_spinner="Melatih model final pada seluruh data...")
# def train_final_model_from_best_scenario(
#     full_df, 
#     use_fs, 
#     use_or, 
#     fs_params, 
#     or_params
# ):
#     """
#     Melatih satu model ARIMAX final pada seluruh dataset historis 
#     berdasarkan skenario terbaik.
#     """

#     df = full_df.copy()
    
#     # --- PREPROCESSING (SCALING) ---
#     # Asumsikan data sudah diimputasi. Kita akan scale di sini.
#     # Fitur waktu sudah dihapus di tahap sebelumnya.
#     df_scaled, scaler = preprocess_scale(df)

#     # --- FEATURE SELECTION (JIKA DIPILIH) ---
#     if use_fs:
#         corr_matrix = df_scaled.corr(method='pearson')
#         corr_with_target = corr_matrix['PM2.5'].abs().drop('PM2.5')
#         features_passed_corr = corr_with_target[corr_with_target >= fs_params['corr_threshold']].index.tolist()
        
#         df_for_vif = df_scaled[features_passed_corr]
#         vif_df = calculate_vif(df_for_vif)
#         final_features = vif_df[vif_df['Skor VIF'] <= fs_params['vif_threshold']]['Fitur'].tolist()
        
#         df_processed = df_scaled[['PM2.5'] + final_features]
#     else:
#         df_processed = df_scaled
#         final_features = df.drop(columns=['PM2.5']).columns.tolist()

#     # --- OUTLIER REMOVAL (JIKA DIPILIH) ---
#     if use_or:
#         df_processed, _, _ = get_outliers_df(df_processed, k=or_params['k_optimal'], silhouette_threshold=or_params['sil_threshold'])

#     # --- PELATIHAN MODEL FINAL ---
#     endog = df_processed['PM2.5']
#     exog = df_processed[final_features]
    
#     # Temukan order terbaik untuk seluruh data
#     final_order = find_best_order_grid_search(endog, max_p=3, max_q=3)
    
#     # Latih model
#     model = SARIMAX(endog, exog=exog, order=final_order)
#     results = model.fit(disp=False)
    
#     # Kembalikan semua artefak penting
#     return results, scaler, final_features, full_df.columns.tolist(), final_order

# def predict_future_values(final_model, scaler, exog_input_df, final_features, original_cols, target_variable='PM2.5'):
#     """
#     Membuat prediksi masa depan dan mengembalikannya ke skala asli.
#     Versi ini memperbaiki pembuatan DataFrame dummy untuk proses scaling.
#     """
    
#     # --- PERBAIKAN LOGIKA UTAMA DI SINI ---

#     # 1. Buat "cetakan" DataFrame yang benar.
#     #    - Buat DataFrame berisi nol dengan SEMUA kolom asli dan indeks yang sama dengan input.
#     #    - Ini memastikan urutan dan nama kolom sama persis seperti yang diharapkan scaler.
#     template_df = pd.DataFrame(0, index=exog_input_df.index, columns=original_cols)

#     # 2. "Tempelkan" nilai input dari pengguna ke dalam cetakan.
#     #    Metode .update() akan mengisi nilai berdasarkan nama kolom yang cocok.
#     template_df.update(exog_input_df)
    
#     # 3. Sekarang, 'template_df' siap untuk di-scale karena strukturnya sudah benar.
#     scaled_array = scaler.transform(template_df)
#     df_scaled = pd.DataFrame(scaled_array, columns=original_cols, index=template_df.index)
    
#     # --- AKHIR PERBAIKAN ---

#     # 4. Ambil hanya kolom fitur yang relevan untuk prediksi
#     exog_for_forecast = df_scaled[final_features]

#     # 5. Buat prediksi (hasil masih dalam skala 0-1)
#     forecast_result = final_model.get_forecast(steps=len(exog_for_forecast), exog=exog_for_forecast)
#     forecast_mean_scaled = forecast_result.predicted_mean
    
#     # 6. Lakukan inverse transform untuk kembali ke skala asli
#     original_predictions = inverse_transform_predictions(forecast_mean_scaled, scaler, original_cols, target_variable)
    
#     return original_predictions, forecast_mean_scaled

# Ganti seluruh fungsi train_final_model_from_best_scenario dengan ini
@st.cache_resource(show_spinner="Melatih model final pada seluruh data...")
def train_final_model_from_best_scenario(
    full_df_imputed, 
    use_fs, use_or, 
    fs_params, or_params
):
    """
    Melatih satu model ARIMAX final pada seluruh dataset historis.
    Versi ini menyimpan 'scaler' dan 'kolom' dengan lebih eksplisit.
    """
    df = full_df_imputed.copy()
    
    # Simpan nama kolom sebelum diproses lebih lanjut
    original_cols_for_scaling = df.columns.tolist()
    
    # --- PREPROCESSING (SCALING) ---
    df_scaled, scaler = preprocess_scale(df)

    # --- FEATURE SELECTION (JIKA DIPILIH) ---
    if use_fs:
        corr_matrix = df_scaled.corr(method='pearson')
        corr_with_target = corr_matrix['PM2.5'].abs().drop('PM2.5')
        features_passed_corr = corr_with_target[corr_with_target >= fs_params['corr_threshold']].index.tolist()
        df_for_vif = df_scaled[features_passed_corr]
        vif_df = calculate_vif(df_for_vif) if not df_for_vif.empty else pd.DataFrame(columns=['Fitur', 'Skor VIF'])
        final_features = vif_df[vif_df['Skor VIF'] <= fs_params['vif_threshold']]['Fitur'].tolist()
        df_processed = df_scaled[['PM2.5'] + final_features]
    else:
        df_processed = df_scaled
        final_features = df.drop(columns=['PM2.5']).columns.tolist()

    # --- OUTLIER REMOVAL (JIKA DIPILIH) ---
    if use_or:
        df_processed, _, _ = get_outliers_df(df_processed, k=or_params['k_optimal'], silhouette_threshold=or_params['sil_threshold'])

    # --- PELATIHAN MODEL FINAL ---
    endog = df_processed['PM2.5']
    exog = df_processed[final_features]
    
    final_order = find_best_order_grid_search(endog, max_p=3, max_q=3)
    model = SARIMAX(endog, exog=exog, order=final_order)
    results = model.fit(disp=False)
    
    # Kembalikan semua artefak penting
    return results, scaler, final_features, original_cols_for_scaling, final_order

# Ganti seluruh fungsi predict_future_values dengan ini
def predict_future_values(final_model, scaler, exog_input_df, final_features, original_cols_for_scaler, target_variable='PM2.5'):
    """
    Membuat prediksi masa depan dan mengembalikannya ke skala asli (versi lebih robust).
    """
    # 1. Buat DataFrame dengan struktur lengkap untuk di-scaling
    #    Isi dengan nol, pastikan urutan kolom sama persis dengan saat scaler dilatih
    prediction_template = pd.DataFrame(np.zeros((len(exog_input_df), len(original_cols_for_scaler))), 
                                       columns=original_cols_for_scaler)
    
    # 2. Isi nilai dari input pengguna ke kolom yang sesuai
    for col in exog_input_df.columns:
        if col in prediction_template.columns:
            prediction_template[col] = exog_input_df[col].values

    # 3. Scale seluruh template DataFrame
    scaled_values = scaler.transform(prediction_template)
    df_scaled = pd.DataFrame(scaled_values, columns=original_cols_for_scaler)
    
    # 4. Ambil kolom eksogen yang sudah di-scale untuk prediksi
    exog_for_forecast = df_scaled[final_features]

    # 5. Buat prediksi (hasil masih dalam skala 0-1)
    forecast_result = final_model.get_forecast(steps=len(exog_for_forecast), exog=exog_for_forecast)
    forecast_mean_scaled = forecast_result.predicted_mean
    
    # 6. Lakukan inverse transform
    original_predictions = inverse_transform_predictions(forecast_mean_scaled.values, scaler, original_cols_for_scaler, target_variable)
    
    return original_predictions, forecast_mean_scaled
