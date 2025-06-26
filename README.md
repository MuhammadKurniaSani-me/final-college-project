# ARIMAX Modeling for Time Series Forecasting with Statistical Significance & Cluster-Based Outlier Removal

This repository contains the source code and analysis for my final college project (*Skripsi*) focusing on time series analysis using an ARIMAX model, with a special emphasis on advanced data preprocessing techniques, including statistical feature selection and cluster-based outlier detection.

The primary goal of this project is to build a robust model for a given time series dataset (e.g., PM2.5 air quality). The methodology ensures that the data is clean, relevant features are selected, and the final model is both statistically sound and accurate.

## 🚀 Live Application

Explore the interactive forecasting model through the deployed Streamlit application:

*(Note: Please replace `YOUR_STREAMLIT_APP_URL_HERE` with the actual URL of your deployed app.)*

## 🛠️ Methodology & Implemented Steps

The project follows a systematic approach to data analysis and modeling. Below is a step-by-step overview of the implemented methods.

### 1. Initial Data Analysis & Visualization

* **Descriptive Statistics:** Initial review of the dataset's statistical properties.
* **Correlation Analysis:** A correlation matrix and heatmap were generated to understand the linear relationships between variables.
* **Data Visualization:** Time series plots, scatter plots (2D and 3D), and pairplots were created to visually inspect trends, seasonality, and relationships between features.

### 2. Feature Selection

* **Pearson Correlation:** Used to identify and select independent variables that have a strong linear relationship with the dependent variable.
* **Variance Inflation Factor (VIF):** Implemented to check for multicollinearity among independent variables, ensuring that the features used in the final model are not redundant.

### 3. Cluster-Based Outlier Detection

A key innovation of this project is the use of K-Means clustering to identify and remove outliers based on how well they fit within the natural data structure.

* **Determining Optimal `k`:**
    * **Elbow Method:** The Sum of Squared Errors (SSE) was calculated for a range of `k` values. The plot showed a significant "elbow" at **`k=2`**, indicating a dominant binary structure in the data.
    * **Silhouette Score:** The average silhouette score was calculated for various `k` values, with the peak score occurring at **`k=2`**.
* **Decision:** After analyzing the trade-offs between variance explanation (Elbow) and cluster cohesion/separation (Silhouette), **`k=2`** was chosen for its clear structural significance and ease of interpretation.
* **Outlier Removal:**
    * The `silhouette_samples` function was used to calculate a silhouette score for every individual data point within the `k=2` clustering result.
    * Observations with a silhouette score below a set threshold (e.g., data <= 0.0) were identified as outliers and removed from the dataset, resulting in a cleaner, more robust dataset for modeling.

### 4. Time Series Modeling with ARIMAX

* **Stationarity Check:** The Augmented Dickey-Fuller (ADF) test was used to check for stationarity in the time series. Differencing (`d=1`) was applied to make the series stationary.
* **Parameter Selection (p, q):** The Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots were analyzed to determine the optimal `p` and `q` parameters.
* **Baseline Model:** A univariate **ARIMA(1,1,1)** model was first built using only the dependent variable to establish a performance baseline.
* **Final Model:** An **ARIMAX (Autoregressive Integrated Moving Average with Exogenous Variables)** model was implemented. This model enhances the baseline ARIMA by including the selected, statistically significant independent variables, leading to a more powerful and accurate forecast.

## 📂 Repository Structure

```
.
├── data/
│   └── sample-data-after-SS.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_outlier_detection_clustering.ipynb
│   └── 03_arimax_modeling.ipynb
├── streamlit_app.py
└── README.md
```
