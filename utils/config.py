"""
Module: utils.config

Global configuration for the project (e.g., data paths, default parameters).
"""
import os

# Root directory of project (one level up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Figures directory
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
# Notebooks directory
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Default resampling frequency
DEFAULT_FREQ = '100ms'
# Default FFT window size (number of samples)
DEFAULT_WINDOW_SIZE = 256
# Default number of FFT features
DEFAULT_N_FFT_FEATURES = 10
# Default historical fetch window (in minutes)
DEFAULT_FETCH_WINDOW_MINUTES = 60