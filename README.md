# Crypto_FFT_ML_Analysis

## Overview

This repository demonstrates a full pipeline for high‑frequency cryptocurrency market analysis using Python. We fetch and preprocess tick‑level trade data from Binance, apply Fast Fourier Transform (FFT) to extract frequency‑domain features, and use machine learning to classify price direction, forecast price changes, and uncover latent regimes via unsupervised clustering. All code is modular, well‑documented, and accompanied by Jupyter notebooks illustrating each step.

## Motivation

Cryptocurrency markets exhibit noisy, non‑stationary behavior, especially at millisecond resolution. Traditional time‑domain approaches often miss subtle periodicities or spectral signatures hidden in the noise. FFT lets us decompose price series into frequency components, revealing rhythmic patterns. Coupled with statistical learning, these spectral features may provide predictive insight or uncover distinct market regimes.

## Requirements

- Python 3.9 or higher
- A virtual environment (venv or conda)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pago-99/Crypto_FFT_ML_Analysis.git
   cd Crypto_FFT_ML_Analysis
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .\.venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Crypto_FFT_ML_Analysis/
├── data_ingestion/            # Fetch public tick and aggregated trade data
├── preprocessing/             # Resampling and order book feature extraction
├── feature_extraction/        # FFT and sliding-window utilities
├── modeling/                  # ML models (classification, regression, clustering, DR)
├── utils/                     # Configuration and plotting helpers
├── notebooks/                 # EDA, FFT, ML, Clustering notebooks
├── data/                      # Saved feature matrices (X_fft.npy)
├── figures/                   # (Optional) exported plot images
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

1. **Data Ingestion** (`data_ingestion/`)

   - `fetch_recent_trades` and `fetch_historical_agg_trades` use Binance’s public REST API without API keys.
   - `LiveTradeStreamer` (optional) streams live trades via WebSocket.

2. **Preprocessing** (`preprocessing/`)

   - `resample_tick_data`: converts irregular tick data into uniform OHLCV bars (100 ms default), forward‑filling prices and zero‑filling volumes.
   - `orderbook_features`: (optional) computes order book imbalance from snapshots.

3. **Feature Extraction** (`feature_extraction/`)

   - `sliding_window`: generates non‑overlapping windows of fixed size (512 samples by default).
   - `compute_fft_features`: extracts top N frequencies and magnitudes from each window via FFT.

4. **Exploratory Data Analysis** (`notebooks/01_exploratory_data_analysis.ipynb`)

   - Visualize raw tick price/volume and resampled bars.
   - Inspect return distributions.

5. **FFT Feature Extraction** (`notebooks/02_fft_feature_extraction.ipynb`)

   - Fetch 10 minutes of historical aggregated trades.
   - Resample to 100 ms bars and generate sliding windows.
   - Compute FFT features and save the feature matrix (`data/X_fft.npy`).

6. **Machine Learning Modeling** (`notebooks/03_ml_modeling.ipynb`)

   - Load `X_fft.npy` and regenerate windows to compute labels:
     - **Classification:** next‑window price direction (up/down).
     - **Regression:** next‑window price change magnitude.
   - Train/test split and fit:
     - `PriceDirectionClassifier` (Logistic Regression)
     - `PriceChangeRegressor` (Linear Regression)
   - Evaluate via confusion matrix and predicted‑vs‑actual plots.

7. **Clustering & Dimensionality Reduction** (`notebooks/04_clustering_and_pca.ipynb`)
   - Apply PCA to reduce `X_fft` to 2D and visualize.
   - Perform K‑Means and DBSCAN on the full feature matrix, plotted in PCA space.
   - Generate a t‑SNE embedding (with `init='pca'` and `learning_rate='auto'`) and visualize K‑Means clusters.

## Running the Notebooks

Open the project folder in VS Code or launch Jupyter from the project root:

```bash
jupyter notebook
# then open files under notebooks/
```

Each notebook includes narrative, code, and inline plots. No additional configuration is needed once dependencies are installed.

## Notes

- The `figures/` directory is reserved for any exported plot images but can remain empty if you only display plots inline.
- Adjust parameters (e.g., `DEFAULT_WINDOW_SIZE`, clustering `n_clusters`, DBSCAN `eps/min_samples`) in `utils/config.py` to explore different resolutions and regimes.

## Author

Yechan Jeong  
Seoul National University Department of Economics
GitHub: [Pago-99](https://github.com/Pago-99)

## License

Released under the MIT License.
