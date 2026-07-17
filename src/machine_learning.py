"""
=========================================================
Spotify Genre Segmentation
Machine Learning Module (Part 1)
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
Project : Spotify Songs Genre Segmentation
Algorithm : K-Means Clustering
=========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# --------------------------------------------------------
# Create Required Directories
# --------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)


# =========================================================
# KMeans Model Class
# =========================================================

class KMeansModel:

    def __init__(self,
                 n_clusters=6,
                 random_state=42):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    # =====================================================
    # Elbow Method
    # =====================================================

    def elbow_method(self,
                     X,
                     max_clusters=10):

        inertia = []

        print("=" * 60)
        print("Finding Optimal Number of Clusters")
        print("=" * 60)

        for k in range(1, max_clusters + 1):

            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=20
            )

            km.fit(X)

            inertia.append(km.inertia_)

            print(f"K={k}  Inertia={km.inertia_:.2f}")

        plt.figure(figsize=(8,5))

        plt.plot(
            range(1, max_clusters + 1),
            inertia,
            marker="o",
            linewidth=2
        )

        plt.xlabel("Number of Clusters")

        plt.ylabel("Inertia")

        plt.title("Elbow Method")

        plt.grid(True)

        plt.savefig(
            "outputs/images/12_elbow_method.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        return inertia

    # =====================================================
    # Train KMeans
    # =====================================================

    def train(self, X):

        print("=" * 60)
        print("Training K-Means Model")
        print("=" * 60)

        self.model = KMeans(

            n_clusters=self.n_clusters,

            random_state=self.random_state,

            n_init=20

        )

        self.model.fit(X)

        print("Training Completed")

        return self.model

    # =====================================================
    # Predict Cluster
    # =====================================================

    def predict(self, X):

        if self.model is None:

            raise Exception("Model is not trained.")

        return self.model.predict(X)

    # =====================================================
    # Assign Clusters to Dataset
    # =====================================================

    def assign_clusters(self,
                        dataframe,
                        X):

        labels = self.predict(X)

        dataframe["Cluster"] = labels

        print("\nClusters Assigned Successfully")

        return dataframe

    # =====================================================
    # Cluster Centers
    # =====================================================

    def cluster_centers(self):

        if self.model is None:

            raise Exception("Train model first.")

        centers = pd.DataFrame(

            self.model.cluster_centers_

        )

        print("\nCluster Centers\n")

        print(centers)

        return centers

    # =====================================================
    # Cluster Size
    # =====================================================

    def cluster_size(self,
                     dataframe):

        print("\nCluster Size\n")

        print(

            dataframe["Cluster"]

            .value_counts()

            .sort_index()

        )

    # =====================================================
    # Save Model
    # =====================================================

    def save_model(
            self,
            filename="models/spotify_kmeans_model.pkl"):

        if self.model is None:

            raise Exception("Train model first.")

        joblib.dump(
            self.model,
            filename
        )

        print(f"\nModel Saved -> {filename}")

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(
            self,
            filename="models/spotify_kmeans_model.pkl"):

        self.model = joblib.load(filename)

        print("Model Loaded Successfully")

        return self.model

    # =====================================================
    # Predict New Song
    # =====================================================

    def predict_single_song(
            self,
            song_features):

        song_features = np.array(song_features).reshape(1,-1)

        cluster = self.model.predict(song_features)

        return cluster[0]

    # =====================================================
    # Save Clustered Dataset
    # =====================================================

    def save_clustered_dataset(
            self,
            dataframe,
            filename="data/processed/spotify_clustered_songs.csv"):

        dataframe.to_csv(

            filename,

            index=False

        )

        print(f"Dataset Saved -> {filename}")
# =====================================================
# Model Evaluation
# =====================================================

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from sklearn.decomposition import PCA


# =====================================================
# Evaluate Model
# =====================================================

def evaluate(self, X):

    if self.model is None:
        raise Exception("Train model first.")

    labels = self.model.labels_

    silhouette = silhouette_score(X, labels)

    calinski = calinski_harabasz_score(X, labels)

    davies = davies_bouldin_score(X, labels)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Silhouette Score          : {silhouette:.4f}")
    print(f"Calinski Harabasz Score   : {calinski:.4f}")
    print(f"Davies Bouldin Score      : {davies:.4f}")

    return {
        "Silhouette Score": silhouette,
        "Calinski Harabasz Score": calinski,
        "Davies Bouldin Score": davies
    }


# =====================================================
# PCA Visualization
# =====================================================

def pca_visualization(self, X):

    if self.model is None:
        raise Exception("Train model first.")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X)

    plt.figure(figsize=(10,8))

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=self.model.labels_,
        cmap="viridis",
        s=18
    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("K-Means Clusters (PCA Projection)")

    plt.colorbar(label="Cluster")

    plt.savefig(
        "outputs/images/14_pca_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("PCA Plot Saved")


# =====================================================
# Cluster Summary
# =====================================================

def cluster_summary(self, dataframe):

    if "Cluster" not in dataframe.columns:

        raise Exception("Cluster column not found.")

    print("=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)

    summary = dataframe.groupby("Cluster").mean(
        numeric_only=True
    )

    print(summary)

    summary.to_csv(
        "outputs/cluster_summary.csv"
    )

    return summary


# =====================================================
# Top Songs Per Cluster
# =====================================================

def top_songs(self,
              dataframe,
              n=10):

    if "Cluster" not in dataframe.columns:
        return

    print("=" * 60)
    print("TOP SONGS FROM EACH CLUSTER")
    print("=" * 60)

    for cluster in sorted(dataframe["Cluster"].unique()):

        print(f"\nCluster {cluster}")

        songs = dataframe[
            dataframe["Cluster"] == cluster
        ].head(n)

        if "track_name" in dataframe.columns:

            print(songs["track_name"].tolist())

        else:

            print(songs.head())


# =====================================================
# Complete Training Pipeline
# =====================================================

def train_pipeline(self,
                   dataframe,
                   X):

    self.elbow_method(X)

    self.train(X)

    dataframe = self.assign_clusters(
        dataframe,
        X
    )

    self.cluster_centers()

    self.cluster_size(dataframe)

    self.evaluate(X)

    self.pca_visualization(X)

    self.cluster_summary(dataframe)

    self.save_model()

    self.save_clustered_dataset(dataframe)

    return dataframe
if __name__ == "__main__":

    from data_preprocessing import DataPreprocessor

    DATASET = "data/raw/spotify_songs.csv"

    processor = DataPreprocessor(DATASET)

    df, X_scaled = processor.preprocessing_pipeline()

    model = KMeansModel(
        n_clusters=6
    )

    clustered_df = model.train_pipeline(
        df,
        X_scaled
    )

    model.top_songs(clustered_df)

    print("\nMachine Learning Pipeline Completed Successfully")
