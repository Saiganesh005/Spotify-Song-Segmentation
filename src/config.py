"""
=========================================================
Spotify Genre Segmentation
Configuration File
=========================================================
"""

import os

# =====================================================
# Project Directories
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

# =====================================================
# Dataset
# =====================================================

DATASET_NAME = "spotify_songs.csv"

DATASET_PATH = os.path.join(
    RAW_DATA_DIR,
    DATASET_NAME
)

# =====================================================
# Random State
# =====================================================

RANDOM_STATE = 42

# =====================================================
# Machine Learning
# =====================================================

N_CLUSTERS = 6

MAX_ITER = 300

N_INIT = 10

# =====================================================
# Deep Learning
# =====================================================

INPUT_DIM = 13

ENCODING_DIM = 8

LEARNING_RATE = 0.001

EPOCHS = 50

BATCH_SIZE = 64

VALIDATION_SPLIT = 0.20

# =====================================================
# Feature Columns
# =====================================================

FEATURE_COLUMNS = [

    "danceability",

    "energy",

    "key",

    "loudness",

    "mode",

    "speechiness",

    "acousticness",

    "instrumentalness",

    "liveness",

    "valence",

    "tempo",

    "duration_ms",

    "time_signature"

]

# =====================================================
# Model Files
# =====================================================

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "spotify_scaler.pkl"
)

KMEANS_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "spotify_kmeans_model.pkl"
)

AUTOENCODER_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "spotify_autoencoder.keras"
)

ENCODER_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "spotify_encoder.keras"
)

DL_CLUSTER_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "spotify_dl_kmeans.pkl"
)

SIMILARITY_MATRIX_FILE = os.path.join(
    MODEL_DIR,
    "similarity_matrix.pkl"
)

# =====================================================
# Output Files
# =====================================================

PROCESSED_DATASET = os.path.join(
    PROCESSED_DATA_DIR,
    "spotify_processed.csv"
)

CLUSTER_DATASET = os.path.join(
    PROCESSED_DATA_DIR,
    "spotify_clustered.csv"
)

EVALUATION_REPORT = os.path.join(
    REPORT_DIR,
    "model_evaluation.csv"
)

# =====================================================
# Create Directories
# =====================================================

DIRECTORIES = [

    DATA_DIR,

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    MODEL_DIR,

    OUTPUT_DIR,

    IMAGE_DIR,

    REPORT_DIR,

    PREDICTION_DIR

]

for directory in DIRECTORIES:

    os.makedirs(directory, exist_ok=True)
