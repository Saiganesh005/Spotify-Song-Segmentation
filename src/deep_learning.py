"""
=========================================================
Spotify Genre Segmentation
Deep Learning Module (Part 1)
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
Project : Spotify Genre Segmentation
Model : Autoencoder
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense
)

from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# Create Required Directories
# ---------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)


# =========================================================
# Deep Learning Model
# =========================================================

class DeepLearningModel:

    def __init__(self):

        self.autoencoder = None
        self.encoder = None
        self.history = None

    # =====================================================
    # Build Autoencoder
    # =====================================================

    def build_autoencoder(self, input_dim):

        """
        Autoencoder Architecture

        Input
            ↓
        64
            ↓
        32
            ↓
        16
            ↓
         8  ← Bottleneck
            ↓
        16
            ↓
        32
            ↓
        64
            ↓
        Output
        """

        print("=" * 60)
        print("Building Autoencoder")
        print("=" * 60)

        input_layer = Input(
            shape=(input_dim,),
            name="Input"
        )

        # ---------------- Encoder ----------------

        x = Dense(
            64,
            activation="relu",
            name="Encoder_64"
        )(input_layer)

        x = Dense(
            32,
            activation="relu",
            name="Encoder_32"
        )(x)

        x = Dense(
            16,
            activation="relu",
            name="Encoder_16"
        )(x)

        bottleneck = Dense(
            8,
            activation="relu",
            name="Bottleneck"
        )(x)

        # ---------------- Decoder ----------------

        x = Dense(
            16,
            activation="relu",
            name="Decoder_16"
        )(bottleneck)

        x = Dense(
            32,
            activation="relu",
            name="Decoder_32"
        )(x)

        x = Dense(
            64,
            activation="relu",
            name="Decoder_64"
        )(x)

        output_layer = Dense(
            input_dim,
            activation="linear",
            name="Output"
        )(x)

        # Complete Model

        self.autoencoder = Model(
            inputs=input_layer,
            outputs=output_layer,
            name="Spotify_Autoencoder"
        )

        # Encoder Model

        self.encoder = Model(
            inputs=input_layer,
            outputs=bottleneck,
            name="Spotify_Encoder"
        )

        print("\nModel Summary\n")

        self.autoencoder.summary()

    # =====================================================
    # Compile Model
    # =====================================================

    def compile_model(self):

        print("=" * 60)
        print("Compiling Autoencoder")
        print("=" * 60)

        self.autoencoder.compile(

            optimizer=Adam(
                learning_rate=0.001
            ),

            loss="mse"

        )

        print("Compilation Completed")

    # =====================================================
    # Train Model
    # =====================================================

    def train(
            self,
            X,
            epochs=50,
            batch_size=64):

        print("=" * 60)
        print("Training Autoencoder")
        print("=" * 60)

        self.history = self.autoencoder.fit(

            X,
            X,

            epochs=epochs,

            batch_size=batch_size,

            validation_split=0.20,

            shuffle=True,

            verbose=1

        )

        print("\nTraining Completed")

        return self.history

    # =====================================================
    # Plot Training Loss
    # =====================================================

    def plot_loss(self):

        if self.history is None:

            print("Train model first.")

            return

        plt.figure(figsize=(8,5))

        plt.plot(

            self.history.history["loss"],

            label="Training Loss",

            linewidth=2

        )

        plt.plot(

            self.history.history["val_loss"],

            label="Validation Loss",

            linewidth=2

        )

        plt.xlabel("Epoch")

        plt.ylabel("Mean Squared Error")

        plt.title("Autoencoder Training Loss")

        plt.legend()

        plt.grid(True)

        plt.savefig(

            "outputs/images/17_training_loss.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.show()

        print("Training Loss Figure Saved")

    # =====================================================
    # Save Autoencoder
    # =====================================================

    def save_model(
            self,
            filename="models/spotify_autoencoder.h5"):

        self.autoencoder.save(filename)

        print(f"\nAutoencoder Saved -> {filename}")

    # =====================================================
    # Save Encoder
    # =====================================================

    def save_encoder(
            self,
            filename="models/spotify_encoder.keras"):

        self.encoder.save(filename)

        print(f"Encoder Saved -> {filename}")
# =====================================================
# Required Imports
# =====================================================

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

import joblib


# =====================================================
# Extract Latent Features
# =====================================================

def extract_features(self, X):

    """
    Generate compressed feature vectors
    using encoder network.
    """

    print("=" * 60)
    print("Extracting Latent Features")
    print("=" * 60)

    encoded_features = self.encoder.predict(
        X,
        verbose=0
    )

    print("Encoded Shape :", encoded_features.shape)

    return encoded_features


# =====================================================
# Deep K-Means Clustering
# =====================================================

def cluster_features(
        self,
        encoded_features,
        n_clusters=6):

    print("=" * 60)
    print("Deep Feature Clustering")
    print("=" * 60)

    self.kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = self.kmeans.fit_predict(
        encoded_features
    )

    print("Clustering Completed")

    return labels


# =====================================================
# Add Cluster Labels
# =====================================================

def assign_clusters(
        self,
        dataframe,
        labels):

    dataframe["DL_Cluster"] = labels

    print("Deep Learning Clusters Assigned")

    return dataframe


# =====================================================
# Evaluate Deep Clustering
# =====================================================

def evaluate(
        self,
        encoded_features,
        labels):

    print("=" * 60)
    print("Deep Learning Evaluation")
    print("=" * 60)

    silhouette = silhouette_score(
        encoded_features,
        labels
    )

    calinski = calinski_harabasz_score(
        encoded_features,
        labels
    )

    davies = davies_bouldin_score(
        encoded_features,
        labels
    )

    print(f"Silhouette Score : {silhouette:.4f}")

    print(f"Calinski Score   : {calinski:.4f}")

    print(f"Davies Score     : {davies:.4f}")

    return {
        "Silhouette": silhouette,
        "Calinski": calinski,
        "Davies": davies
    }


# =====================================================
# PCA Visualization
# =====================================================

def pca_visualization(
        self,
        encoded_features,
        labels):

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        encoded_features
    )

    plt.figure(figsize=(10,8))

    plt.scatter(

        reduced[:,0],

        reduced[:,1],

        c=labels,

        cmap="viridis",

        s=20

    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("Deep Learning Clusters")

    plt.colorbar(label="Cluster")

    plt.savefig(
        "outputs/images/18_deep_learning_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("PCA Figure Saved")


# =====================================================
# Reconstruction Error
# =====================================================

def reconstruction_error(self, X):

    reconstructed = self.autoencoder.predict(
        X,
        verbose=0
    )

    mse = np.mean(
        np.square(X - reconstructed),
        axis=1
    )

    plt.figure(figsize=(8,5))

    plt.hist(
        mse,
        bins=40
    )

    plt.xlabel("Reconstruction Error")

    plt.ylabel("Frequency")

    plt.title("Autoencoder Reconstruction Error")

    plt.savefig(
        "outputs/images/23_reconstruction_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return mse


# =====================================================
# Save KMeans Model
# =====================================================

def save_cluster_model(
        self,
        filename="models/spotify_dl_kmeans.pkl"):

    joblib.dump(
        self.kmeans,
        filename
    )

    print(f"Saved : {filename}")


# =====================================================
# Save Encoded Features
# =====================================================

def save_encoded_features(
        self,
        encoded_features,
        filename="data/processed/encoded_features.npy"):

    np.save(
        filename,
        encoded_features
    )

    print(f"Saved : {filename}")


# =====================================================
# Cluster Summary
# =====================================================

def cluster_summary(
        self,
        dataframe):

    summary = dataframe.groupby(

        "DL_Cluster"

    ).mean(

        numeric_only=True

    )

    print(summary)

    summary.to_csv(

        "outputs/deep_cluster_summary.csv"

    )

    return summary
# =====================================================
# Predict Cluster for New Song
# =====================================================

def predict_song(self, song_features):

    """
    Predict cluster for a new Spotify song.
    """

    song_features = np.array(song_features).reshape(1, -1)

    encoded = self.encoder.predict(
        song_features,
        verbose=0
    )

    cluster = self.kmeans.predict(encoded)

    return int(cluster[0])


# =====================================================
# Load Saved Models
# =====================================================

def load_models(
        self,
        autoencoder_path="models/spotify_autoencoder.h5",
        encoder_path="models/spotify_encoder.keras",
        kmeans_path="models/spotify_dl_kmeans.pkl"):

    from tensorflow.keras.models import load_model

    print("=" * 60)
    print("Loading Saved Models")
    print("=" * 60)

    self.autoencoder = load_model(
        autoencoder_path
    )

    self.encoder = load_model(
        encoder_path
    )

    self.kmeans = joblib.load(
        kmeans_path
    )

    print("Models Loaded Successfully")


# =====================================================
# Complete Deep Learning Pipeline
# =====================================================

def train_pipeline(
        self,
        dataframe,
        X,
        epochs=50,
        batch_size=64,
        n_clusters=6):

    print("=" * 60)
    print("STARTING DEEP LEARNING PIPELINE")
    print("=" * 60)

    # ----------------------------
    # Build Network
    # ----------------------------

    self.build_autoencoder(
        input_dim=X.shape[1]
    )

    self.compile_model()

    self.train(
        X,
        epochs=epochs,
        batch_size=batch_size
    )

    self.plot_loss()

    # ----------------------------
    # Feature Extraction
    # ----------------------------

    encoded = self.extract_features(X)

    labels = self.cluster_features(
        encoded,
        n_clusters=n_clusters
    )

    dataframe = self.assign_clusters(
        dataframe,
        labels
    )

    self.evaluate(
        encoded,
        labels
    )

    self.pca_visualization(
        encoded,
        labels
    )

    self.reconstruction_error(
        X
    )

    self.cluster_summary(
        dataframe
    )

    # ----------------------------
    # Save Everything
    # ----------------------------

    self.save_model()

    self.save_encoder()

    self.save_cluster_model()

    self.save_encoded_features(
        encoded
    )

    print("=" * 60)
    print("Deep Learning Pipeline Completed")
    print("=" * 60)

    return dataframe


# =====================================================
# Display Model Information
# =====================================================

def model_information(self):

    print("=" * 60)
    print("AUTOENCODER INFORMATION")
    print("=" * 60)

    print(self.autoencoder.summary())

    print("\nEncoder Summary\n")

    print(self.encoder.summary())
# =====================================================
# Main Function
# =====================================================

if __name__ == "__main__":

    from data_preprocessing import DataPreprocessor

    DATASET = "data/raw/spotify_songs.csv"

    # -----------------------------------------
    # Load Dataset
    # -----------------------------------------

    processor = DataPreprocessor(DATASET)

    dataframe, X_scaled = processor.preprocessing_pipeline()

    # -----------------------------------------
    # Deep Learning Model
    # -----------------------------------------

    dl_model = DeepLearningModel()

    clustered_dataframe = dl_model.train_pipeline(

        dataframe,

        X_scaled,

        epochs=50,

        batch_size=64,

        n_clusters=6

    )

    # -----------------------------------------
    # Example Prediction
    # -----------------------------------------

    sample = X_scaled[0]

    cluster = dl_model.predict_song(sample)

    print("\nPredicted Cluster :", cluster)

    print("\nDeep Learning Module Executed Successfully.")
  
