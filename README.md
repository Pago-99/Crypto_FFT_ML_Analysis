# Crypto_FFT_ML_Analysis

## Overview

This repository demonstrates a full pipeline for high-frequency cryptocurrency market analysis using Python. We fetch and preprocess tick-level trade data from Binance, apply Fast Fourier Transform (FFT) to extract frequency-domain features, and use machine learning to classify price direction, forecast price changes, and uncover latent regimes via unsupervised clustering. All code is modular, well-documented, and accompanied by Jupyter notebooks illustrating each step.

## Motivation

Cryptocurrency markets exhibit noisy, non-stationary behavior, especially at millisecond resolution. Traditional time-domain approaches often miss subtle periodicities or spectral signatures hidden in the noise. FFT lets us decompose price series into frequency components, revealing rhythmic patterns. Coupled with statistical learning, these spectral features may provide predictive insight or uncover distinct market regimes.

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
   .\.venv\Scripts\activate  # Windows
   ```
3. **Install runtime dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install the package in editable mode**
   ```bash
   pip install -e .
   ```

This setup ensures both notebook execution and module imports work correctly.

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
2. **Preprocessing** (`preprocessing/`)
3. **Feature Extraction** (`feature_extraction/`)
4. **Exploratory Data Analysis** (`notebooks/01_exploratory_data_analysis.ipynb`)
5. **FFT Feature Extraction** (`notebooks/02_fft_feature_extraction.ipynb`)
6. **Machine Learning Modeling** (`notebooks/03_ml_modeling.ipynb`)
7. **Clustering & Dimensionality Reduction** (`notebooks/04_clustering_and_pca.ipynb`)

Refer to each notebook for detailed code, plots, and explanations.

## Running the Notebooks

Once dependencies are installed and the package is in editable mode, launch Jupyter Notebook from the project root:
```bash
jupyter notebook
```
This will open a browser, where you can navigate into the `notebooks/` directory and run each `.ipynb` file interactively.

## Author

Yechan Jeong  
Seoul National University  
GitHub: [Pago-99](https://github.com/Pago-99)

## License

Released under the MIT License.
