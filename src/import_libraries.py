"""
=========================================================
Spotify Genre Segmentation
Import Libraries Module
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

# =====================================================
# Standard Libraries
# =====================================================

import os
import sys
import warnings
import pickle
import random
import time

# =====================================================
# Data Handling
# =====================================================

import numpy as np
import pandas as pd

# =====================================================
# Data Visualization
# =====================================================

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Machine Learning
# =====================================================

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)

from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Deep Learning
# =====================================================

import tensorflow as tf

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import (
    Input,
    Dense
)

from tensorflow.keras.optimizers import Adam

# =====================================================
# Save / Load Models
# =====================================================

import joblib

# =====================================================
# Plot Settings
# =====================================================

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (10, 6)

plt.rcParams["axes.titlesize"] = 14

plt.rcParams["axes.labelsize"] = 12

# =====================================================
# Random Seed
# =====================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)

# =====================================================
# Create Project Directories
# =====================================================

PROJECT_FOLDERS = [

    "data",

    "data/raw",

    "data/processed",

    "models",

    "outputs",

    "outputs/images",

    "outputs/reports",

    "outputs/predictions"

]

for folder in PROJECT_FOLDERS:

    os.makedirs(folder, exist_ok=True)

# =====================================================
# Exported Modules
# =====================================================

__all__ = [

    # Standard Libraries
    "os",
    "sys",
    "warnings",
    "pickle",
    "random",
    "time",

    # Data Handling
    "np",
    "pd",

    # Visualization
    "plt",
    "sns",

    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",

    # ML
    "KMeans",
    "PCA",
    "train_test_split",

    # Metrics
    "silhouette_score",
    "silhouette_samples",
    "calinski_harabasz_score",
    "davies_bouldin_score",

    # Recommendation
    "cosine_similarity",

    # Deep Learning
    "tf",
    "Model",
    "load_model",
    "Input",
    "Dense",
    "Adam",

    # Save / Load
    "joblib",

    # Constants
    "RANDOM_STATE"
]
