# utils.py (Versi Final dan Bersih)

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
import joblib
import os

# Plotting
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

# Statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Scipy
import scipy.spatial.distance


# --- Konstanta ---
WIND_DIRECTION_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135,
    'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
    'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

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
    # Definisikan fitur waktu yang akan dikecualikan dari analisis korelasi
    time_features = ['year', 'month', 'day', 'hour']
    
    # Buat salinan DataFrame dan hapus fitur waktu
    # errors='ignore' akan mencegah error jika kolom tersebut sudah tidak ada
    df_for_corr = df.drop(columns=time_features, errors='ignore')

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


# --- FUNGSI STYLING ---

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


# --- Pemuatan Data ---
@st.cache_data
def load_station_data(station_name):
    """
    Memuat data, membuat DatetimeIndex, dan menghapus kolom waktu asli.
    """
    try:
        file_path = f'./datas/locations/PRSA_Data_{station_name}_20130301-20170228.csv'
        df = pd.read_csv(file_path)
        if 'No' in df.columns:
            df = df.drop(columns=['No'])

        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index('datetime', inplace=True)
        df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"File data untuk stasiun {station_name} tidak ditemukan.")
        return None

# --- Pipeline Prapemrosesan ---
def preprocess_encoding(df, station_name, station_code_map):
    """Meng-encode kolom kategorikal."""
    df_encoded = df.copy()
    if 'wd' in df_encoded.columns and df_encoded['wd'].dtype == 'object':
        mode_val = df_encoded['wd'].mode()[0]
        df_encoded['wd'].fillna(mode_val, inplace=True)
        df_encoded['wd'] = df_encoded['wd'].map(WIND_DIRECTION_MAP)
    
    if 'station' not in df_encoded.columns:
        df_encoded['station'] = station_name
    if df_encoded['station'].dtype == 'object':
        df_encoded['station'] = df_encoded['station'].map(station_code_map)
    return df_encoded

def preprocess_impute(df):
    """Mengisi nilai yang hilang menggunakan interpolasi linear."""
    return df.interpolate(method='linear', limit_direction='both', axis=0)

def preprocess_scale(df):
    """Menskalakan data menggunakan MinMaxScaler."""
    scaler = MinMaxScaler()
    df_numeric = df.select_dtypes(include=np.number)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), index=df_numeric.index, columns=df_numeric.columns)
    return df_scaled, scaler

# --- Logika Outlier Removal ---
def kmeans_mahalanobis(df, k, max_iterations=100):
    """Menjalankan K-Means menggunakan Mahalanobis distance."""
    data = df.to_numpy()
    n_samples, n_features = data.shape
    try:
        cov_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.identity(n_features)
    
    kmeans_init = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    kmeans_init.fit(data)
    centroids = kmeans_init.cluster_centers_

    for _ in range(max_iterations):
        distances = np.array([[scipy.spatial.distance.mahalanobis(point, centroids[i], inv_cov_matrix) for i in range(k)] for point in data])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids): break
        centroids = new_centroids
        
    return labels, centroids, 0 # Inertia tidak dihitung untuk simplicity

def get_outliers_df(df, k, silhouette_threshold=0.0):
    """Mengidentifikasi outlier menggunakan K-Means Mahalanobis."""
    df_result = df.copy()
    labels, _, _ = kmeans_mahalanobis(df_result, k)
    df_result['cluster'] = labels
    df_result['silhouette_score'] = silhouette_samples(df_result.drop('cluster', axis=1), df_result['cluster'])
    inliers = df_result[df_result['silhouette_score'] >= silhouette_threshold].copy()
    outliers = df_result[df_result['silhouette_score'] < silhouette_threshold].copy()
    inliers.drop(columns=['cluster', 'silhouette_score'], inplace=True)
    return inliers, outliers, df_result

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


def find_best_order_grid_search(series, max_p=3, max_q=3):
    """
    Mencari order ARIMA (p,d,q) terbaik menggunakan grid search dan uji ADF.
    Fungsi ini murni untuk komputasi dan mengembalikan hasilnya beserta log.
    """
    logs = []
    
    # 1. Menentukan orde diferensiasi (d) dengan Uji ADF
    adf_test = adfuller(series, autolag='AIC')
    p_value = adf_test[1]
    d = 1 if p_value > 0.05 else 0
    logs.append(f"    - Hasil Uji ADF p-value: {p_value:.3f} -> Menentukan d={d}")
    logs.append(f"    - Memulai Grid Search untuk (p,q)...")

    # 2. Grid Search untuk p dan q untuk menemukan AIC terendah
    best_aic = float('inf')
    best_order = None
    
    warnings.filterwarnings("ignore") # Abaikan warning konvergensi dari statsmodels
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0: continue
            try:
                model = SARIMAX(series, order=(p, d, q))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except Exception:
                continue # Lanjutkan jika ada kombinasi yang error
    warnings.filterwarnings("default") # Kembalikan warning seperti semula

    # Fallback jika tidak ada model yang berhasil
    if best_order is None: best_order = (1, d, 1)
    
    logs.append(f"    - Order terbaik ditemukan: {best_order} (AIC: {best_aic:.2f})")
        
    return best_order, logs

def _evaluate_single_station(raw_df, use_fs, use_or, fs_params, or_params, station_name):
    """
    Fungsi helper yang bersih untuk menjalankan pipeline lengkap untuk satu stasiun.
    Versi final ini tidak lagi mengandung elemen UI.
    """
    target_variable = 'PM2.5'
    all_rmse_scores, all_r2_scores = [], []
    station_logs = [f"--- Log untuk Stasiun: {station_name} ---"]

    # Tahap 1: Preprocessing
    if 'station' not in raw_df.columns: raw_df['station'] = station_name
    station_code_map = {station_name: 0}
    encoded_df = preprocess_encoding(raw_df, station_name, station_code_map)
    imputed_df = preprocess_impute(encoded_df)
    scaled_df, _ = preprocess_scale(imputed_df)
    station_logs.append("✅ Prapemrosesan Selesai (Encoding, Imputasi, Scaling).")

    # Tahap 2: Seleksi Fitur
    if use_fs:
        corr_matrix = scaled_df.corr(method='pearson')
        corr_with_target = corr_matrix[target_variable].abs().drop(target_variable)
        features_passed_corr = corr_with_target[corr_with_target >= fs_params['corr_threshold']].index.tolist()
        df_for_vif = scaled_df[features_passed_corr]
        vif_df = calculate_vif(df_for_vif) if not df_for_vif.empty else pd.DataFrame(columns=['Fitur', 'Skor VIF'])
        final_features = vif_df[vif_df['Skor VIF'] <= fs_params['vif_threshold']]['Fitur'].tolist()
    else:
        final_features = scaled_df.drop(columns=[target_variable]).columns.tolist()
    df_scenario = scaled_df[[target_variable] + final_features].copy()
    station_logs.append(f"Kolom yang dipakai: {df_scenario.columns.tolist()}")

    # Tahap 3: Outlier Removal (sebelum split)
    if use_or:
        station_logs.append("    - Menjalankan Outlier Removal pada keseluruhan data...")
        df_scenario, outliers_df, _ = get_outliers_df(df_scenario, k=or_params['k_optimal'], silhouette_threshold=or_params['sil_threshold'])
        station_logs.append(f"    - Ditemukan & dihapus {len(outliers_df)} outlier. Sisa data: {len(df_scenario)} baris.")

    # Tahap 4: Evaluasi
    n_repetitions, train_percent, test_percent = 10, 0.6, 0.1
    N = len(df_scenario)
    train_samples = math.floor(train_percent * N)
    test_samples = math.floor(test_percent * N)
    station_logs.append(f"Total observasi setelah prapemrosesan (N): {N}")
    station_logs.append(f"Sampel latih per jendela: {train_samples} ({train_percent*100:.0f}% dari N)")
    station_logs.append(f"Sampel uji per jendela: {test_samples} ({test_percent*100:.0f}% dari N)")
    
    splits = list(rep_holdout_splitter(df_scenario, n_repetitions, train_percent, test_percent))
    
    for i, (train_df, test_df) in enumerate(splits):

        station_logs.append(f"\n--- Repetisi {i + 1} ---")
        station_logs.append(f"  Periode Latih: {train_df.index[0].date()} hingga {train_df.index[-1].date()}, Panjang: {len(train_df)}")
        station_logs.append(f"  Periode Uji:  {test_df.index[0].date()} hingga {test_df.index[-1].date()}, Panjang: {len(test_df)}")

        endog_train = train_df[target_variable]
        
        if len(endog_train) < 30: # Pengecekan keamanan
            station_logs.append(f"    - ⚠️ PERINGATAN: Data latih terlalu sedikit ({len(endog_train)} baris). Melewati repetisi ini.")
            continue

        order, order_logs = find_best_order_grid_search(endog_train, max_p=3, max_q=3)
        station_logs.extend(order_logs)

        exog_train = train_df[final_features]
        exog_test = test_df[final_features]
        endog_test = test_df[target_variable]
        
        model = SARIMAX(endog_train, exog=exog_train, order=order)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=len(endog_test), exog=exog_test)
        forecast_mean = forecast.predicted_mean
        
        rmse = np.sqrt(mean_squared_error(endog_test, forecast_mean))
        all_rmse_scores.append(rmse)
        r2 = r2_score(endog_test, forecast_mean)
        all_r2_scores.append(r2)
        
    return {"avg_rmse": np.mean(all_rmse_scores) if all_rmse_scores else None, 
            "avg_r2": np.mean(all_r2_scores) if all_r2_scores else None, 
            "logs": station_logs}

@st.cache_data # Cache tetap di sini karena ini adalah fungsi utama yang berat
def run_evaluation_for_all_stations(scenario_name, use_fs, use_or):

    """
    Fungsi utama yang murni untuk komputasi.
    Ia menjalankan evaluasi untuk semua stasiun dan mengembalikan hasilnya
    beserta semua log yang terkumpul.
    """

    STATION_NAMES = [
    'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 
    'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'
    ]
    
    fs_params = {'corr_threshold': 0.3, 'vif_threshold': 10.0}
    or_params = {'k_optimal': 2, 'sil_threshold': 0.3}
    
    results_per_station = []
    full_log_data = [] # List untuk mengumpulkan semua log dari semua stasiun

    # Loop melalui setiap stasiun
    for i, station in enumerate(STATION_NAMES):

        # 1. Muat data untuk stasiun saat ini ke dalam variabel 'raw_df'
        raw_df = load_station_data(station)
        
        # 2. Lanjutkan hanya jika data berhasil dimuat
        if raw_df is None: 
            continue
        
        # 3. Panggil fungsi helper dengan 'raw_df' yang sudah terdefinisi
        station_metrics = _evaluate_single_station(
            raw_df=raw_df,  # <-- Gunakan variabel yang sudah ada
            use_fs=use_fs, 
            use_or=use_or, 
            fs_params=fs_params, 
            or_params=or_params, 
            station_name=station
        )
        
        # Kumpulkan hasil log
        #full_log_data.extend(station_metrics['logs'])

        if station_metrics: # Hanya tambahkan jika evaluasi berhasil
            results_per_station.append({
                "Stasiun": station, 
                "Rata-rata-RMSE": station_metrics['avg_rmse'],
                "Rata-rata-R": station_metrics['avg_r2']
            })
            full_log_data.extend(station_metrics['logs'])
        
    # Kembalikan DUA objek: DataFrame hasil dan List log
    return pd.DataFrame(results_per_station), full_log_data

@st.cache_resource(show_spinner="Melatih model final pada seluruh data...")
def train_final_model_from_best_scenario(
    full_df, 
    use_fs, use_or, 
    fs_params, or_params,
    station_name
):
    """
    Melatih satu model ARIMAX final pada seluruh dataset historis.
    Versi ini menyimpan 'scaler' dan 'kolom' dengan lebih eksplisit.
    """

    if 'station' not in full_df.columns: full_df['station'] = station_name
    station_code_map = {station_name: 0}
    encoded_df = preprocess_encoding(full_df, station_name, station_code_map)
    imputed_df = preprocess_impute(encoded_df)
    
    df = imputed_df.copy()
    
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
    
    final_order, _ = find_best_order_grid_search(endog, max_p=3, max_q=3)
    model = SARIMAX(endog, exog=exog, order=final_order)
    results = model.fit(disp=False)
    
    # Kembalikan semua artefak penting 
    return results, scaler, final_features, original_cols_for_scaling, final_order

def load_prediction_artifacts(artifacts_path='../models/prediction_artifacts.joblib'):
    """
    Memuat dictionary artefak yang disimpan (scaler, daftar fitur, dll.).
    """
    try:
        artifacts = joblib.load(artifacts_path)
        # Muat model statsmodels secara terpisah menggunakan path dari artefak
        artifacts['model'] = SARIMAXResults.load(artifacts['model_path'])
        return artifacts
    except FileNotFoundError:
        return None

def predict_future_values(final_model, scaler, exog_input_df, final_features, target_variable='PM2.5'):
    """
    Membuat prediksi masa depan dan mengembalikannya ke skala asli (versi lebih robust).
    """
    # --- PERBAIKAN LOGIKA UTAMA DI SINI ---
    
    # 1. Ambil daftar & urutan kolom yang benar langsung dari objek scaler
    required_cols = scaler.feature_names_in_
    
    # 2. Buat "cetakan" DataFrame yang sempurna menggunakan daftar kolom dari scaler
    prediction_template = pd.DataFrame(0.0, 
                                       index=exog_input_df.index, 
                                       columns=required_cols)

    # 3. "Tempelkan" nilai input dari pengguna ke dalam cetakan.
    prediction_template.update(exog_input_df)
    
    # 4. Sekarang, 'prediction_template' dijamin cocok dengan scaler
    scaled_values = scaler.transform(prediction_template)
    df_scaled = pd.DataFrame(scaled_values, columns=required_cols, index=prediction_template.index)
    
    # --- AKHIR PERBAIKAN ---
    # 5. Ambil hanya kolom fitur yang relevan untuk prediksi
    exog_for_forecast = df_scaled[final_features]

    # 6. Buat prediksi (hasil masih dalam skala 0-1)
    forecast_result = final_model.get_forecast(steps=len(exog_for_forecast), exog=exog_for_forecast)
    forecast_mean_scaled = forecast_result.predicted_mean
    
    # 7. Lakukan inverse transform (required_cols sama dengan original_cols_for_scaler)
    original_predictions = inverse_transform_predictions(
        forecast_mean_scaled.values, scaler, required_cols, target_variable
    )
    
    return original_predictions, forecast_mean_scaled

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
    
    # --- PERBAIKAN DI SINI ---
    # 1. Konversi numpy array 'all_feature_names' menjadi sebuah Python list
    all_feature_names_list = list(all_feature_names)
    
    # 2. Cari indeks pada list tersebut, bukan pada numpy array
    try:
        target_idx = all_feature_names_list.index(target_variable)
    except ValueError:
        # Fallback jika nama target tidak ditemukan
        return np.array([None] * len(scaled_predictions))
    # --- AKHIR PERBAIKAN ---

    original_scale_predictions = inversed_df[:, target_idx]
    
    return original_scale_predictions
