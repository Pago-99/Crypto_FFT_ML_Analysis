"""
Module: utils.plotting

Helper functions for plotting time-series, FFT spectra, and model diagnostics.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_time_series(df: pd.DataFrame, column: str, title: str = None, save_path: str = None):
    """
    Plot a time-series column from DataFrame.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df[column])
    plt.title(title or f'Time Series: {column}')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_fft_spectrum(series: pd.Series, sample_rate: float = 1.0, title: str = None, save_path: str = None):
    """
    Plot the FFT amplitude spectrum of a series.
    """
    y = series.values.astype(float) - np.mean(series.values)
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0/sample_rate)
    mags = np.abs(fft_vals)

    plt.figure(figsize=(12, 4))
    plt.stem(freqs, mags, markerfmt=' ', basefmt=' ')
    plt.title(title or 'FFT Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list = None, save_path: str = None):
    """
    Plot a confusion matrix for classification results.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot predicted vs actual values for regression.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Regression: Predicted vs Actual')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_clusters(X_2d: np.ndarray, labels: np.ndarray, title: str = None, save_path: str = None):
    """
    Scatter plot of 2D reduced data with cluster labels.
    """
    plt.figure(figsize=(8,6))
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(X_2d[mask,0], X_2d[mask,1], label=f'Cluster {lab}', alpha=0.6)
    plt.legend()
    plt.title(title or 'Cluster Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()